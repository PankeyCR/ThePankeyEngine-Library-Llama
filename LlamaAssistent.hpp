#pragma once

#include <string>

#if defined(pankey_Log) && (defined(LlamaAssistent_Log) || defined(pankey_Global_Log) || defined(pankey_Library_Llama_Log))
	#include "Logger_status.hpp"
	#define LlamaAssistentLog(status,method,mns) pankey_Log(status,"LlamaAssistent",method,mns)
#else
	#define LlamaAssistentLog(status,method,mns)
#endif

namespace pankey{

	namespace Library{

		namespace Llama{

			class LlamaAssistent{
				public:
					LlamaAssistent(){}
					~LlamaAssistent(){}
					
					void conversationLength(int a_size){
						m_context.reserve(a_size);
					}

					void setChatTemplate(std::string a_model){
						if(a_model == "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M"){
							m_start = "<|im_start|>";    
                            m_end = "<|im_end|>";        
                            m_user_name = "user";          
                            m_assistent_name = "assistant";
                            m_think = "";      
                            // m_think = "<think>";      
						}
					}

					void instructions(std::string a_intructions){
						m_context = a_intructions;
					}

					void instructions(){
						m_context = "This is a convesation between a user and his assistent";
					}

					void addUserPrompt(const std::string a_prompt){
						m_context += m_start;
						m_context += m_user_name;
						m_context += "\n";
						m_context += a_prompt;
						m_context += m_end;
						m_context += "\n";
					}

					void addAssistentStart(){
						m_context += m_start;
						m_context += m_assistent_name;
						m_context += "\n";
						m_context += m_think;
						m_context += "\n";
					}

					std::string getContext(){
						return m_context;
					}

					void addAssistentResponse(const std::string a_response){
						m_context += a_response;
						m_context += m_end;
						m_context += "\n";
					}

				protected:
					std::string m_start = "<Start_Message>";
					std::string m_end = "<End_Message>";
					std::string m_user_name = "User:";
					std::string m_assistent_name = "Assistent:";
					std::string m_think = "<Think>";

					std::string m_context;
			};

		}

	}

}