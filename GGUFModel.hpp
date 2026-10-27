#pragma once

#include "llama.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

#if defined(pankey_Log) && (defined(GGUFModel_Log) || defined(pankey_Global_Log) || defined(pankey_ArduinoIO_Log))
	#include "Logger_status.hpp"
	#define GGUFModelLog(status,method,mns) pankey_Log(status,"GGUFModel",method,mns)
#else
	#define GGUFModelLog(status,method,mns)
#endif

namespace pankey{

	namespace Library{

		namespace Llama{

			struct GGUFModel{

			inline static bool s_backend_loaded = false;

			std::shared_ptr<llama_model>   m_model;
			std::shared_ptr<llama_context> m_context;

			void loadAbsolutePath(const std::string& a_path){

				if(!s_backend_loaded){
					ggml_backend_load_all();
					s_backend_loaded = true;
				}

				unload();

				std::filesystem::path i_model_path =
					(std::filesystem::absolute(a_path)).lexically_normal();

				if(!std::filesystem::exists(i_model_path)){
					GGUFModelLog(0, "load", ("model not found: " + i_model_path.string()).c_str());
					return;
				}

				llama_model_params i_model_params =
					llama_model_default_params();

				llama_model* i_model =
					llama_model_load_from_file(
						i_model_path.string().c_str(),
						i_model_params
					);

				if(i_model == nullptr){
					GGUFModelLog(0, "load", "failed to load model");
					return;
				}

				m_model = std::shared_ptr<llama_model>(
					i_model,
					llama_model_free
				);
			}

			void load(const std::string& a_path){

				if(!s_backend_loaded){
					ggml_backend_load_all();
					s_backend_loaded = true;
				}

				unload();

				std::filesystem::path i_model_path =
					(std::filesystem::current_path() / a_path).lexically_normal();

				if(!std::filesystem::exists(i_model_path)){
					GGUFModelLog(0, "load", ("model not found: " + i_model_path.string()).c_str());
					return;
				}

				llama_model_params i_model_params =
					llama_model_default_params();

				llama_model* i_model =
					llama_model_load_from_file(
						i_model_path.string().c_str(),
						i_model_params
					);

				if(i_model == nullptr){
					GGUFModelLog(0, "load", "failed to load model");
					return;
				}

				m_model = std::shared_ptr<llama_model>(
					i_model,
					llama_model_free
				);
			}

			void context(
				uint32_t a_n_ctx = 4096,
				uint32_t a_n_batch = 512
			){

				if(!m_model){
					GGUFModelLog(0, "context", "model is not loaded");
					return;
				}

				m_context.reset();

				llama_context_params i_context_params =
					llama_context_default_params();

				i_context_params.n_ctx   = a_n_ctx;
				i_context_params.n_batch = a_n_batch;

				llama_context* i_context =
					llama_init_from_model(
						m_model.get(),
						i_context_params
					);

				if(i_context == nullptr){
					GGUFModelLog(0, "context", "failed to create context");
					return;
				}

				m_context = std::shared_ptr<llama_context>(
					i_context,
					llama_free
				);
			}

			const llama_vocab* vocab() const{

				if(!m_model){
					return nullptr;
				}

				return llama_model_get_vocab(
					m_model.get()
				);
			}

			llama_model* model() const{
				return m_model.get();
			}

			llama_context* context() const{
				return m_context.get();
			}

			bool loaded() const{
				return m_model != nullptr;
			}

			bool initialized() const{
				return m_context != nullptr;
			}

			virtual void unload(){

				m_context.reset();
				m_model.reset();
			}

			virtual ~GGUFModel(){
				unload();
			}
		};
		}

	}

}