#pragma once

#include "LLM.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <optional>

namespace pankey {

    namespace Library {

        namespace Llama {

            class AsyncLLMWorker {
            public:
                AsyncLLMWorker() = default;

                ~AsyncLLMWorker() {
                    stop();
                }

                // Initialize the worker and launch the background thread
                void load_start(const std::string& a_model_path, 
                        uint32_t a_n_ctx = 4096, 
                        uint32_t a_n_batch = 512) 
                {
                    stop(); // Ensure any existing thread is cleaned up
                    m_llm.loadAbsolutePath(a_model_path);

                    m_running = true;
                    m_workerThread = std::jthread([this, a_model_path, a_n_ctx, a_n_batch](std::stop_token stopToken) {
                        // Load model and set context inside the worker thread
                        if (m_llm.loaded()) {
                            m_llm.context(a_n_ctx, a_n_batch);
                        }

                        workerLoop(stopToken);
                    });
                }

                void start(const std::string& a_model_path, 
                        uint32_t a_n_ctx = 4096, 
                        uint32_t a_n_batch = 512) 
                {
                    stop(); // Ensure any existing thread is cleaned up

                    m_running = true;
                    m_workerThread = std::jthread([this, a_model_path, a_n_ctx, a_n_batch](std::stop_token stopToken) {
                        // Load model and set context inside the worker thread
                        m_llm.loadAbsolutePath(a_model_path);
                        if (m_llm.loaded()) {
                            m_llm.context(a_n_ctx, a_n_batch);
                        }

                        workerLoop(stopToken);
                    });
                }

                // Stop thread and release resources
                void stop() {
                    m_running = false;
                    m_cv.notify_all();
                    if (m_workerThread.joinable()) {
                        m_workerThread.request_stop();
                        m_workerThread.join();
                    }
                }
                
                std::string getChatTemplate() {
                    std::lock_guard<std::mutex> lock(m_mutex);

                    return m_llm.getChatTemplate();
                }
                
                std::string getMetaData() {
                    std::lock_guard<std::mutex> lock(m_mutex);

                    return m_llm.getMetaData();
                }

                // Submit a prompt to process (Non-blocking call for your state loop)
                bool sendPrompt(const std::string& a_prompt, size_t a_max_tokens = 128) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (m_hasPendingPrompt || m_isProcessing) {
                        return false; // Busy processing an active prompt
                    }

                    m_pendingPrompt = a_prompt;
                    m_maxTokens = a_max_tokens;
                    m_hasPendingPrompt = true;
                    m_cv.notify_one();
                    return true;
                }
                
                bool startNewConversation() {
                    std::lock_guard<std::mutex> lock(m_mutex);

                    if (m_hasPendingPrompt || m_isProcessing) {
                        return false;
                    }

                    m_resetContext = true;
                    m_hasPendingPrompt = true;

                    m_cv.notify_one();
                    return true;
                }

                // Poll complete text output once generation finishes (Non-blocking)
                std::optional<std::string> pollResponse() {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (m_hasNewResponse) {
                        m_hasNewResponse = false;
                        return std::move(m_latestResponse);
                    }
                    return std::nullopt;
                }

                // Poll individual tokens as they generate if you want token-by-token streaming
                std::optional<std::string> pollToken() {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (!m_tokenQueue.empty()) {
                        std::string token = std::move(m_tokenQueue.front());
                        m_tokenQueue.pop();
                        return token;
                    }
                    return std::nullopt;
                }

                bool isBusy() const {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    return m_hasPendingPrompt || m_isProcessing;
                }

                bool startResponse(){
                    std::lock_guard<std::mutex> lock(m_mutex);
                    bool i_start = m_start;
                    if(i_start){
                        m_start = false;
                    }
                    return i_start;
                }

                bool endResponse(){
                    std::lock_guard<std::mutex> lock(m_mutex);
                    bool i_end = m_end;
                    if(i_end){
                        m_end = false;
                    }
                    return i_end;
                }

            private:
                void workerLoop(std::stop_token stopToken) {
                    while (!stopToken.stop_requested() && m_running) {
                        std::string currentPrompt;
                        size_t maxTokens = 128;

                        {
                            std::unique_lock<std::mutex> lock(m_mutex);
                            m_cv.wait(lock, [this, &stopToken] {
                                return m_hasPendingPrompt || stopToken.stop_requested() || !m_running;
                            });

                            if (stopToken.stop_requested() || !m_running) break;

                            currentPrompt = m_pendingPrompt;
                            maxTokens = m_maxTokens;
                            m_hasPendingPrompt = false;
                            m_isProcessing = true;
                        }

                        if (m_llm.initialized()) {
                            // Decode prompt tokens into model context
                            {
                                std::lock_guard<std::mutex> lock(m_mutex);
                                if (m_resetContext) {
                                    m_llm.resetContext();
                                    m_resetContext = false;
                                }
                            }
                            if (m_llm.prompt(currentPrompt)) {
                                {
                                    std::lock_guard<std::mutex> lock(m_mutex);
                                    m_start = true;
                                }
                                // Generate response token-by-token while populating stream queue
                                std::string fullResponse = generateAndStream(maxTokens, stopToken);

                                std::lock_guard<std::mutex> lock(m_mutex);
                                m_latestResponse = std::move(fullResponse);
                                m_hasNewResponse = true;
                                m_end = true;
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(m_mutex);
                            m_isProcessing = false;
                        }
                    }

                    m_llm.unload();
                }

                // Custom internal generation loop to enable streaming without locking the super loop
                std::string generateAndStream(size_t a_max_tokens, std::stop_token& stopToken) {
                    const llama_vocab* i_vocab = m_llm.vocab();
                    std::string fullResponse;

                    for (size_t i = 0; i < a_max_tokens; ++i) {
                        if (stopToken.stop_requested()) break;

                        llama_token i_token = llama_sampler_sample(
                            m_llm.m_sampler.get(), 
                            m_llm.m_context.get(), 
                            -1
                        );

                        if (llama_vocab_is_eog(i_vocab, i_token)) {
                            break;
                        }

                        llama_sampler_accept(m_llm.m_sampler.get(), i_token);

                        char i_piece[256];
                        const int32_t i_piece_size = llama_token_to_piece(
                            i_vocab, 
                            i_token, 
                            i_piece, 
                            sizeof(i_piece), 
                            0, 
                            true
                        );

                        if (i_piece_size > 0) {
                            std::string pieceStr(i_piece, static_cast<size_t>(i_piece_size));
                            fullResponse += pieceStr;

                            // Push token into stream queue for UI/State polling
                            std::lock_guard<std::mutex> lock(m_mutex);
                            m_tokenQueue.push(std::move(pieceStr));
                        }

                        llama_batch i_batch = llama_batch_get_one(&i_token, 1);
                        if (llama_decode(m_llm.m_context.get(), i_batch) != 0) {
                            break;
                        }
                    }

                    return fullResponse;
                }

                LLM m_llm;

                mutable std::mutex m_mutex;
                std::condition_variable_any m_cv;
                std::jthread m_workerThread;

                std::string m_pendingPrompt;
                size_t m_maxTokens = 128;

                std::string m_latestResponse;
                std::queue<std::string> m_tokenQueue;

                bool m_hasPendingPrompt = false;
                bool m_hasNewResponse = false;
                bool m_isProcessing = false;
                bool m_running = false;
                
                bool m_start = false;
                bool m_end = false;

                bool m_resetContext = false;
            };

        } // namespace Llama

    } // namespace Library
    
} // namespace pankey