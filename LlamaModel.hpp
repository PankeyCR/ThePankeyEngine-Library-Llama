#pragma once

#include "TemplateSavableAppState.hpp"
#include "ByteArray.hpp"
#include "LLM.hpp"
#include "AsyncLLMWorker.hpp"

#if defined(pankey_Log) && (defined(LlamaModel_Log) || defined(pankey_Global_Log) || defined(pankey_Library_Llama_Log))
	#include "Logger_status.hpp"
	#define LlamaModelLog(status,method,mns) pankey_Log(status,"LlamaModel",method,mns)
#else
	#define LlamaModelLog(status,method,mns)
#endif

namespace pankey{

	namespace Library{

		namespace Llama{			

			struct LlamaStartModel{};
			struct LlamaEndModel{};

			struct LlamaModel{
				template<typename P, typename A>
				using BASESTATE_TYPE = pankey::Runtime::Policy::TemplateSavableAppState<P, A>;

				using STATE_TYPE = AsyncLLMWorker;
				
				using INITIALIZEMODEL_TYPE = LlamaModel;
				using UPDATEMODEL_TYPE = LlamaModel;

				using WRITEMODEL_TYPE = LlamaModel;
				using EXPORTER_TYPE = LlamaModel;

				using READMODEL_TYPE = LlamaModel;
				using IMPORTER_TYPE = LlamaModel;
			};

			template<class S, class A>
			void initializeState(LlamaModel a_model, S& a_state, A& a_ip){
				LlamaModelLog(pankey_Log_StartMethod, "initializeState", "");
				
				LlamaModelLog(pankey_Log_EndMethod, "initializeState", "");
			}

			template<class S, class A, class... Args>
			void updateState(LlamaModel a_model, S& a_state, A& a_app, Args... a_values){
				LlamaModelLog(pankey_Log_StartMethod, "updateState",  "");
				if(a_state.startResponse()){
					auto& i_events = a_app.getEvents();
					i_events.template runEvent<LlamaStartModel,A&>(a_app);
				}
				if(auto fullResponse = a_state.pollToken()){
					std::string i_text = *fullResponse;
					auto& i_events = a_app.getEvents();
					i_events.template runEvent<LlamaModel,A&,const std::string&>(a_app, i_text);
				}
				if(a_state.endResponse()){
					auto& i_events = a_app.getEvents();
					i_events.template runEvent<LlamaEndModel,A&>(a_app);
				}
				LlamaModelLog(pankey_Log_EndMethod, "updateState",  "");
			}

		}

	}

}