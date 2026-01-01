from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Optional
import json
import os

from rag.vector_store import VectorStoreManager
from models.schemas import FAQResponse, SourceDocument


class FAQRagSystem:
    def __init__(self, data_path: str, vector_db_path: str, collection_name: str):
        self.data_path = data_path
        self.vector_store_manager = VectorStoreManager(vector_db_path, collection_name)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2
        )
        self.qa_chain = None
        self.intents_data = None
        self.retriever = None

    def load_intents_data(self) -> List[Dict]:
        """Load intents data from JSON file - supports both formats"""
        print(f"Loading intents data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            print(f"Warning: Intents file not found at {self.data_path}")
            return []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'intents' in data:
                self.intents_data = data.get('intents', [])
            elif isinstance(data, list):
                self.intents_data = data
            elif isinstance(data, dict) and not 'intents' in data:
                self.intents_data = self._convert_qa_to_intents(data)
            else:
                print("Warning: Unexpected intents.json format")
                self.intents_data = []
        
        print(f"Loaded {len(self.intents_data)} intents")
        return self.intents_data

    def _convert_qa_to_intents(self, qa_data: Dict) -> List[Dict]:
        """Convert Q&A format to intents format"""
        intents = []
        
        for category, qa_pairs in qa_data.items():
            if not isinstance(qa_pairs, list):
                continue
                
            for idx, qa_pair in enumerate(qa_pairs):
                if not isinstance(qa_pair, dict):
                    continue
                
                question = qa_pair.get('q', '')
                answer = qa_pair.get('a', '')
                
                if question and answer:
                    tag = f"{category}_{idx}"
                    
                    intent = {
                        'tag': tag,
                        'category': category,
                        'patterns': [question],
                        'responses': [answer]
                    }
                    intents.append(intent)
        
        print(f"Converted {len(intents)} Q&A pairs to intents format")
        return intents

    def prepare_documents(self) -> List[Document]:
        """Convert intents data to LangChain documents with enhanced content"""
        documents = []
        
        for intent in self.intents_data:
            tag = intent.get('tag', '')
            category = intent.get('category', tag)
            patterns = intent.get('patterns', [])
            responses = intent.get('responses', [])
            
            pattern_text = " | ".join(patterns)
            response_text = " | ".join(responses)
            
            keywords = self._extract_keywords(category, patterns, responses)
            keyword_text = " ".join(keywords)
            
            content = f"Category: {category}\nQuestions: {pattern_text}\nAnswer: {response_text}\nKeywords: {keyword_text}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "tag": tag,
                    "category": category,
                    "patterns": json.dumps(patterns),
                    "responses": json.dumps(responses),
                    "source": "intents.json"
                }
            )
            documents.append(doc)
        
        print(f"Prepared {len(documents)} documents for vector store")
        return documents

    def _extract_keywords(self, category: str, patterns: List[str], responses: List[str]) -> List[str]:
        """Extract keywords for better semantic matching"""
        keywords = [category]
        
        keyword_map = {
            'covid': ['coronavirus', 'covid-19', 'pandemic', 'mask', 'protocol', 'safety'],
            'insurance': ['coverage', 'provider', 'billing', 'payment'],
            'hours': ['schedule', 'open', 'closed', 'time', 'operation'],
            'location': ['address', 'where', 'parking', 'directions'],
            'cancellation': ['cancel', 'reschedule', 'policy'],
            'appointment': ['visit', 'booking', 'schedule']
        }
        
        text = ' '.join(patterns + responses).lower()
        for key, values in keyword_map.items():
            if key in text:
                keywords.extend(values)
        
        return list(set(keywords))

    def initialize(self, force_recreate: bool = False) -> None:
        """Initialize the RAG system"""
        self.load_intents_data()
        
        if not self.intents_data or len(self.intents_data) == 0:
            print("Warning: No intents data loaded. System will have no knowledge base.")
            return
        
        vector_store_exists = os.path.exists(self.vector_store_manager.persist_directory)
        
        if force_recreate or not vector_store_exists:
            print("Creating new vector store...")
            documents = self.prepare_documents()
            if documents:
                self.vector_store_manager.create_vector_store(documents)
            else:
                print("Warning: No documents to create vector store")
                return
        else:
            print("Loading existing vector store...")
            self.vector_store_manager.load_vector_store()
        
        self._create_qa_chain()
        
        doc_count = self.vector_store_manager.get_collection_count()
        print(f"RAG system initialized with {doc_count} documents")

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_qa_chain(self) -> None:
        """Create the QA chain using modern LCEL syntax"""
        template = """You are a helpful medical clinic assistant. Answer questions concisely and directly.

Context:
{context}

Question: {question}

Instructions:
- Keep answers brief and to the point (2-4 sentences maximum)
- Use bullet points for lists or multiple items
- If context is insufficient, give a short response directing them to contact the clinic
- Be friendly but concise , always professional redirect to clinic for complex queries

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        if not self.vector_store_manager.vector_store:
            print("Warning: Vector store not available, cannot create retriever")
            return
        
        self.retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": 5}  
        )

        self.qa_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _is_conversational_query(self, question: str) -> tuple[bool, str]:
        """Check if query is conversational and return appropriate response"""
        question_lower = question.lower().strip()
        
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        if any(question_lower == g or question_lower.startswith(f"{g} ") or question_lower.startswith(f"{g},") for g in greetings):
            return True, "Hello! I'm your clinic assistant. I can help you with information about our clinic hours, location, insurance, billing, appointment policies, and visit preparation. What would you like to know?"
        
        status_queries = ['how are you', 'how r you', 'how do you do', 'whats up', "what's up", 'how are things']
        if any(q in question_lower for q in status_queries):
            return True, "I'm doing great, thank you for asking! I'm here to help answer your questions about our clinic. What information can I provide for you today?"
        
        thanks = ['thank you', 'thanks', 'thx', 'appreciate it', 'thank u']
        if any(question_lower == t or question_lower.startswith(f"{t} ") or question_lower.startswith(f"{t}!") for t in thanks):
            return True, "You're very welcome! Feel free to ask if you have any other questions about our clinic services or policies."
        
        goodbye = ['bye', 'goodbye', 'see you', 'take care', 'good bye', 'later', 'have a good day']
        if any(question_lower == g or question_lower.startswith(f"{g} ") or question_lower.startswith(f"{g}!") for g in goodbye):
            return True, "Goodbye! Have a wonderful day. Feel free to reach out anytime you have questions about our clinic."
        
        return False, ""

    def _convert_distance_to_similarity(self, distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score.
        ChromaDB with cosine distance returns values where:
        - 0 = identical (most similar)
        - 2 = opposite (least similar)
        
        We convert to 0-1 scale where 1 = most similar
        """
    
        if distance < 0:
            similarity = 1.0 - (abs(distance) / 2)
        else:
            similarity = 1.0 - (distance / 2)
        
        return max(0.0, min(1.0, similarity))

    def ask(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> FAQResponse:
        """Process a question and return an answer"""
        
        is_conversational, conv_response = self._is_conversational_query(question)
        if is_conversational:
            return FAQResponse(
                answer=conv_response,
                sources=[],
                confidence="high",
                follow_up_suggestions=[
                    "What are your clinic hours?",
                    "What insurance providers do you accept?"
                ]
            )
        
        if not self.qa_chain or not self.retriever:
            return FAQResponse(
                answer="I'm not properly initialized yet. Please make sure the knowledge base is loaded.",
                sources=[],
                confidence="low",
                follow_up_suggestions=[]
            )
        
        doc_count = self.vector_store_manager.get_collection_count()
        if doc_count == 0:
            return FAQResponse(
                answer="I don't have any information loaded yet. Please contact the clinic administrator.",
                sources=[],
                confidence="low",
                follow_up_suggestions=[]
            )
        
        try:
            contextualized_question = question
            if conversation_history and len(conversation_history) > 0:
                recent_context = conversation_history[-3:]
                context_str = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in recent_context
                ])
                contextualized_question = f"Previous conversation:\n{context_str}\n\nCurrent question: {question}"
            
            source_docs = self.retriever.invoke(contextualized_question)
            
            answer = self.qa_chain.invoke(contextualized_question)
            
            sources = []
            search_results = self.vector_store_manager.similarity_search(question, k=5)
            
            for doc in source_docs[:3]:
                score = 0.0
                for search_doc, search_distance in search_results:
                    if search_doc.page_content == doc.page_content:
                        score = self._convert_distance_to_similarity(search_distance)
                        break
                
                sources.append(SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    tag=doc.metadata.get('tag', 'unknown'),
                    relevance_score=round(score, 3)
                ))
            
            if sources and sources[0].relevance_score > 0.6:
                confidence = "high"
            elif sources and sources[0].relevance_score > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            print(f"Top relevance score: {sources[0].relevance_score if sources else 0}")
            print(f"Confidence: {confidence}")
            
            follow_ups = self._generate_follow_ups(question, source_docs)
            
            return FAQResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                follow_up_suggestions=follow_ups
            )
            
        except Exception as e:
            print(f"Error in ask method: {str(e)}")
            import traceback
            traceback.print_exc()
            return FAQResponse(
                answer=f"I encountered an error processing your question. Please try rephrasing or contact our clinic directly.",
                sources=[],
                confidence="low",
                follow_up_suggestions=[]
            )

    def _generate_follow_ups(
        self,
        question: str,
        source_docs: List[Document]
    ) -> List[str]:
        """Generate contextual follow-up questions"""
        follow_ups = []
        
        if not source_docs:
            return ["What are your clinic hours?", "What insurance do you accept?"]
        
        categories = list(set([
            doc.metadata.get('category', '') 
            for doc in source_docs 
            if doc.metadata.get('category')
        ]))
        
        category_suggestions = {
            'clinic_details': [
                "What are your clinic hours?",
                "Where is the clinic located?",
                "Is parking available?"
            ],
            'insurance_billing': [
                "What insurance providers do you accept?",
                "What payment methods can I use?",
                "Can you explain your billing policies?"
            ],
            'visit_preparation': [
                "What documents do I need for my first visit?",
                "What should I bring to my appointment?"
            ],
            'policies': [
                "What's your cancellation policy?",
                "What are your COVID-19 protocols?",
                "What happens if I'm late?"
            ]
        }
        
        for category in categories[:2]:
            if category in category_suggestions:
                suggestions = category_suggestions[category]
                for suggestion in suggestions[:2]:
                    if suggestion.lower() not in question.lower():
                        follow_ups.append(suggestion)
        
        follow_ups = list(dict.fromkeys(follow_ups))
        return follow_ups[:2] if follow_ups else ["What else can I help you with?"]

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vector store"""
        if not self.vector_store_manager.vector_store:
            print("Creating new vector store with documents...")
            self.vector_store_manager.create_vector_store(documents)
        else:
            print(f"Adding {len(documents)} documents to existing vector store...")
            self.vector_store_manager.add_documents(documents)
        
        self._create_qa_chain()
        print(f"Vector store now has {self.vector_store_manager.get_collection_count()} documents")

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_documents": self.vector_store_manager.get_collection_count(),
            "total_intents": len(self.intents_data) if self.intents_data else 0,
            "retriever_initialized": self.retriever is not None,
            "qa_chain_initialized": self.qa_chain is not None
        }