#pragma once

#include "GGUFModel.hpp"

#if defined(pankey_Log) && (defined(LLM_Log) || defined(pankey_Global_Log) || defined(pankey_ArduinoIO_Log))
	#include "Logger_status.hpp"
	#define LLMLog(status,method,mns) pankey_Log(status,"LLM",method,mns)
#else
	#define LLMLog(status,method,mns)
#endif

namespace pankey{

	namespace Library{

		namespace Llama{
			
			struct LLM : public GGUFModel{
				
				std::shared_ptr<llama_sampler> m_sampler;

				void context(
					uint32_t a_n_ctx = 4096,
					uint32_t a_n_batch = 512
				){

					if(!m_model){
						LLMLog(0, "context", "model is not loaded");
						return;
					}

					m_sampler.reset();
					m_context.reset();

					llama_context_params i_context_params =
						llama_context_default_params();

					i_context_params.n_ctx = a_n_ctx;
					i_context_params.n_batch = a_n_batch;

					llama_context* i_context =
						llama_init_from_model(
							m_model.get(),
							i_context_params
						);

					if(i_context == nullptr){
						LLMLog(0, "context", "failed to create context");
						return;
					}

					m_context = std::shared_ptr<llama_context>(
						i_context,
						llama_free
					);

					llama_sampler* i_sampler =
						llama_sampler_chain_init(
							llama_sampler_chain_default_params()
						);

					if(i_sampler == nullptr){
						LLMLog(0, "context", "failed to create sampler");
						m_context.reset();
						return;
					}

					llama_sampler_chain_add(
						i_sampler,
						llama_sampler_init_greedy()
					);

					m_sampler = std::shared_ptr<llama_sampler>(
						i_sampler,
						llama_sampler_free
					);
				}
				
				void resetContext() {
					if (!m_context) {
						LLMLog(0, "resetContext", "context is not initialized");
						return;
					}

					// Clear the KV cache / sequence memory.
					llama_memory_clear(
						llama_get_memory(m_context.get()),
						true
					);

					// Reset sampler state for the new inference.
					if (m_sampler) {
						llama_sampler_reset(m_sampler.get());
					}
				}

				std::string getChatTemplate() const {
					if (m_model.get() == nullptr) {
						return std::string();
					}

					const char* tmpl = llama_model_chat_template(
						m_model.get(),
						nullptr
					);

					if (tmpl == nullptr) {
						return std::string();
					}

					return std::string(tmpl);
				}

				std::string getMetaData() const {
					std::string key = "tokenizer.chat_template";
					std::string chatTemplate;

					int32_t len = llama_model_meta_val_str(
						m_model.get(),
						key.c_str(),
						nullptr,
						0
					);

					std::cout << "Template length: " << len << '\n';

					if (len > 0) {
						std::vector<char> buffer(static_cast<size_t>(len) + 1, '\0');

						llama_model_meta_val_str(
							m_model.get(),
							key.c_str(),
							buffer.data(),
							buffer.size()
						);

						chatTemplate = buffer.data();
					}
					return chatTemplate;
				}

				bool prompt(const std::string& a_prompt){

					if(!m_context || !m_sampler){
						LLMLog(0, "prompt", "context is not initialized");
						return false;
					}

					const llama_vocab* i_vocab =
						llama_model_get_vocab(
							m_model.get()
						);

					const std::string i_prompt = a_prompt;

					const bool i_add_special = true;
					const bool i_parse_special = true;

					const int32_t i_prompt_token_count = -llama_tokenize(
						i_vocab,
						i_prompt.c_str(),
						static_cast<int32_t>(i_prompt.size()),
						nullptr,
						0,
						i_add_special,
						i_parse_special
					);

					if(i_prompt_token_count <= 0){
						LLMLog(0, "prompt", "failed to measure prompt");
						return false;
					}

					std::vector<llama_token> i_prompt_tokens(
						static_cast<size_t>(i_prompt_token_count)
					);

					if(llama_tokenize(
						i_vocab,
						i_prompt.c_str(),
						static_cast<int32_t>(i_prompt.size()),
						i_prompt_tokens.data(),
						static_cast<int32_t>(i_prompt_tokens.size()),
						i_add_special,
						i_parse_special
					) < 0){
						LLMLog(0, "prompt", "failed to tokenize prompt");
						return false;
					}

					for(size_t i = 0; i < i_prompt_tokens.size(); ++i){

						llama_batch i_batch =
							llama_batch_get_one(
								&i_prompt_tokens[i],
								1
							);

						if(llama_decode(
							m_context.get(),
							i_batch
						) != 0){
							LLMLog(0, "prompt", "failed to decode prompt");
							return false;
						}
					}

					return true;
				}

				std::string generate(size_t a_max_tokens = 128){

					if(!m_context || !m_sampler){
						LLMLog(0, "generate", "context is not initialized");
						return "";
					}

					const llama_vocab* i_vocab =
						llama_model_get_vocab(
							m_model.get()
						);

					std::string i_response;

					for(size_t i = 0; i < a_max_tokens; ++i){

						llama_token i_token =
							llama_sampler_sample(
								m_sampler.get(),
								m_context.get(),
								-1
							);

						if(llama_vocab_is_eog(
							i_vocab,
							i_token
						)){
							break;
						}

						llama_sampler_accept(
							m_sampler.get(),
							i_token
						);

						char i_piece[256];

						const int32_t i_piece_size =
							llama_token_to_piece(
								i_vocab,
								i_token,
								i_piece,
								sizeof(i_piece),
								0,
								true
							);

						if(i_piece_size > 0){
							i_response.append(
								i_piece,
								static_cast<size_t>(i_piece_size)
							);
						}

						llama_batch i_batch =
							llama_batch_get_one(
								&i_token,
								1
							);

						if(llama_decode(
							m_context.get(),
							i_batch
						) != 0){
							LLMLog(0, "generate", "decode failed");
							break;
						}
					}

					return std::move(i_response);
				}

				void unload(){

					m_sampler.reset();
					m_context.reset();
					m_model.reset();
				}
			};
			
		}

	}

}