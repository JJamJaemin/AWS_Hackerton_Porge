import openai
import streamlit as st
import json
import boto3
# For Claude 3, use BedrockChat instead of Bedrock
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# AWS Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

openai.api_key = "sk-Fj3RajMzYgXqbShy8W_SzJvsBO2eTWiulo7d-XXrYJT3BlbkFJNAQIU3uc2alZpQEz13S2kGc6WOwwh3twbADY11R5MA"


print("up")

# 기본 상태 설정
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_purpose" not in st.session_state:
    st.session_state.selected_purpose = None

def create_prompt(purpose):
    print("create_prompt")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{purpose}에 맞춘 프롬프트를 작성해줘. 예를들어 storytelling이면 창의적인 글쓰기와 스토리텔링에 열정을 가진 AI느낌이고, comfort이면 사용자 감정에 대해서 공감해주어서 사용자의 기분을 좋게 만들어 주는 친구 AI이고, lesson_plan이면 주어진 주제에 대해서 60분 동안 강의 할 수 있는 계획을 짜 주는 AI이게 만드는 프롬프트를 작성해줘. 너가 말한 답변은 또 다른 GPT에게 Instructions안에 넣어줄 내용이야 그 양식에 맞게 한글로 대답해. 예시는 들지 마"},
            {"role": "user", "content": purpose}
        ],
        temperature = 0.1
    )
    prompt_text = response.choices[0].message['content'].strip()
    return prompt_text
    
# 모델 선택 및 프롬프트 수정 함수
# 프롬프트 수정 함수
def get_prompt_with_purpose(prompt, purpose):
    global modified_prompt
    print("get_prompt_with_purpose")
    if purpose == "storytelling":
        modified_prompt = create_prompt(purpose)
        return f"{modified_prompt} : {prompt}"
    elif purpose == "comfort":
        modified_prompt = create_prompt(purpose)
        return f"{modified_prompt} : {prompt}"
    elif purpose == "lesson_plan":
        modified_prompt = create_prompt(purpose)
        return f"{modified_prompt} : {prompt}"
    else:
        return prompt

def get_response(prompt, purpose):
    try:
        modified_prompt = get_prompt_with_purpose(prompt, purpose)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": modified_prompt}],
                    }
                ],
            }
        )

        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body,
        )
        response_body = json.loads(response.get("body").read())
        output_text = response_body["content"][0]["text"]
        return output_text
    except Exception as e:
        print(e)
        return "Sorry, there was an error processing your request."

# Streamlit UI 구성
st.title("AI Assistant")

# 사용자의 목적 선택 버튼
purpose_col1, purpose_col2, purpose_col3 = st.columns(3)

with purpose_col1:
    if st.button("Storytelling"):
        st.session_state.selected_purpose = "storytelling"

with purpose_col2:
    if st.button("Comfort"):
        st.session_state.selected_purpose = "comfort"

with purpose_col3:
    if st.button("Lesson Plan"):
        st.session_state.selected_purpose = "lesson_plan"

# 현재 선택된 목적 표시
if st.session_state.selected_purpose:
    st.write(f"Selected purpose: {st.session_state.selected_purpose}")

# 채팅 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
if st.session_state.selected_purpose == "storytelling":
    # Bedrock 클라이언트 설정
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    
    # Claude 3.5 파라미터 설정
    model_kwargs =  { 
        "max_tokens": 1000,
        "temperature": 0.01,
        "top_p": 0.01,
    }
    
    # Bedrock LLM 설정
    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs=model_kwargs,
        streaming=True
    )
    
    # Streamlit 앱 설정
    st.title("스토리보드")
    
    # Streamlit 채팅 메시지 히스토리 설정
    message_history = StreamlitChatMessageHistory(key="chat_messages")
    
    modified_prompt = ""
    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", modified_prompt),
            MessagesPlaceholder(variable_name="message_history"),
            ("human", "{query}"),
        ]
    )
    
    # 대화 체인 설정
    chain_with_history = RunnableWithMessageHistory(
        prompt | llm,
        lambda session_id: message_history,  # 항상 이전 대화를 리턴
        input_messages_key="query",
        history_messages_key="message_history",
    )
    
    # 채팅 인터페이스
    for msg in message_history.messages:
        st.chat_message(msg.type).write(msg.content)
    
    # 사용자 입력 처리
    if query := st.chat_input("Message Bedrock..."):
        st.chat_message("human").write(query)
    
        # chain이 호출되면 새 메시지가 자동으로 StreamlitChatMessageHistory에 저장됨
        config = {"configurable": {"session_id": "any"}}
        response_stream = chain_with_history.stream({"query": query},config=config)
        st.chat_message("ai").write_stream(response_stream)
        
        
        
        
        
        
        
if st.session_state.selected_purpose == "comfort":
    # Streamlit app settings
    st.title("감정을 공감해주는 AI")
    st.caption("해결방법만 제시해주던 GPT가 공감을?")
    
    # Initialize message history for Streamlit
    message_history = StreamlitChatMessageHistory(key="chat_messages")
    
    purpose = st.session_state.selected_purpose
    
    # 사용자 입력 및 응답 처리
    prompt = st.text_input("Message Bedrock...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
    res = get_response(prompt, purpose)
    print("res")
    print(res)

    st.chat_message("ai").write(res)

    # Display chat messages from the history
    for msg in message_history.messages:
        st.chat_message(msg.type).write(msg.content)
    
    def get_streaming_gpt_response(prompt_ai):
        global modified_prompt
        print("get_streaming_gpt_response")
        try:
            messages = [
                {
                    "role": "system",
                    "content": modified_prompt
                },
                {
                    "role": "user",
                    "content": prompt_ai
                }
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                stream=True  # Stream the response for real-time updates
            )
            
            print("chunk")
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"!!!!!!!!!!!!{e}")
            
            # Process user input and OpenAI API call
    if prompt_ai := st.chat_input("Enter your context for interview questions..."):
        model_output = get_response(prompt, purpose)
        # Display the user's input in the chat
        st.chat_message("user").write(prompt_ai)
        st.chat_message("ai").write_stream(get_streaming_gpt_response(prompt_ai))
        
        
        
        
        
        
        
        
        
if st.session_state.selected_purpose == "lesson_plan":
    # Streamlit app settings
    st.title("강의 계획서를 짜주는 AI")
    st.caption("과목만 적으면?")
    
    # Initialize message history for Streamlit
    message_history = StreamlitChatMessageHistory(key="chat_messages")
    
    purpose = st.session_state.selected_purpose
    
    # 사용자 입력 및 응답 처리
    prompt = st.text_input("Message Bedrock...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
    res = get_response(prompt, purpose)
    print("lesson res")
    print(res)

    st.chat_message("ai").write(res)
    
    # Display chat messages from the history
    for msg in message_history.messages:
        st.chat_message(msg.type).write(msg.content)
    
    def get_streaming_gpt_response2(prompt_ai):
        global modified_prompt
        print("get_streaming_gpt_response2")
        try:
            messages = [
                {
                    "role": "system",
                    "content": modified_prompt
                },
                {
                    "role": "user",
                    "content": prompt_ai
                }
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                stream=True  # Stream the response for real-time updates
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"!!!!!!!!!!!!{e}")
            
            # Process user input and OpenAI API call
    if prompt_ai := st.chat_input("Enter your context for interview questions..."):
        # Display the user's input in the chat
        st.chat_message("user").write(prompt_ai)
        st.chat_message("ai").write_stream(get_streaming_gpt_response2(prompt_ai))
               