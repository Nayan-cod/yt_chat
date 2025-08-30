import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
# Function to extract video_id from YouTube link
def extract_video_id(url: str) -> str:
    """
    Extracts YouTube video ID from a given URL.
    Works with formats like:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


# UI Title
st.title("🎥 YouTube Q&A with LangChain")

# User Inputs
youtube_url = st.text_input("Enter YouTube Video Link:", 
                            value="https://www.youtube.com/watch?v=PGUdWfB8nLg")
question = st.text_input("Enter your question:", value="Summarize the video")
word_limit = st.number_input("Enter number of words for content:", 
                             min_value=50, max_value=20000, value=300)

# Language selection for transcript
lang_choice = st.selectbox(
    "Choose transcript language:",
    options=["en", "hi", "gu", "fr", "es", "de"],
    format_func=lambda x: {
        "en": "English",
        "hi": "Hindi",
        "gu": "Gujarati",
        "fr": "French",
        "es": "Spanish",
        "de": "German"
    }.get(x, x)
)

if st.button("Get Answer"):
    try:
        # Step 1a - Extract video_id
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
            st.stop()

        # Step 1b - Fetch transcript in selected language
        try:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=[lang_choice])
        except NoTranscriptFound:
            st.error(f"No transcript found in {lang_choice}. Try another language.")
            st.stop()

        transcript = " ".join(chunk.text for chunk in transcript_list)
        total_words = len(transcript.split())
        st.write(f"Transcript contains ~{total_words} words in {lang_choice}.")

        # Step 1c - Text Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # Step 1d - Embeddings + FAISS (force CPU to avoid meta tensor error)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Step 2 - LLM
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="conversational",
            temperature=0.2,
            max_new_tokens=2000
        )
        model = ChatHuggingFace(llm=llm)

        # Step 3 - Prompt
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Answer ONLY from the provided transcript context. 
            If the context is insufficient, just say you don't know.

            {context}

            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        # Step 4 - Retrieval + Prompt + Answer
        retrieved_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        final_prompt = prompt.invoke({"context": context_text, "question": question})
        answer = model.invoke([HumanMessage(content=final_prompt.to_string())])

        # Show output
        st.subheader("Answer:")
        st.write(answer.content)

        # Word limit check
        if total_words < word_limit:
            st.warning("⚠️ Video content is less than requested words. Showing available content only.")

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
