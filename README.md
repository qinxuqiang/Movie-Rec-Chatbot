# Movie-Rec-Chatbot
Chatbot providing personal movie recommendations

# Movie Recommendation Chatbot - Design Document

## 1. Executive Summary

**Project Name:** ReelTalk - AI-Powered Movie Recommendation Chatbot  
**Version:** 1.0  
**Date:** July 2025  
**Purpose:** An intelligent conversational system that provides personalized movie recommendations through natural language interactions.

### Key Features
- Natural language processing for movie preferences
- Semantic search with vector embeddings
- Fuzzy matching for names and genres
- Multi-stage conversation flow
- Modern web interface with Gradio

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  Movie Agent    │    │ Recommendation  │
│                 │◄──►│                 │◄──►│    Engine       │
│ - Chat Interface│    │ - Conversation  │    │ - Semantic      │
│ - State Mgmt    │    │ - Preference    │    │ - Filtering     │
│ - User Input    │    │ - Response Gen  │    │ - Ranking       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Data Layer    │
                       │                 │
                       │ - FAISS Vector  │
                       │ - Movie CSV     │
                       │ - Embeddings    │
                       └─────────────────┘
```

### 2.2 Component Breakdown

#### Frontend Layer (Gradio UI)
- **Chat Interface**: Real-time conversation display
- **Input Processing**: User message handling
- **State Management**: Conversation and preference tracking
- **Styling**: Modern dark theme with animations

#### Agent Layer (MovieRecommendationAgent)
- **Conversation Manager**: Multi-stage dialog flow
- **Preference Extractor**: NLP-based preference parsing
- **Response Generator**: Context-aware response creation
- **Workflow Controller**: State transition management

#### Recommendation Engine (ImprovedMovieRecommendationEngine)
- **Semantic Search**: Vector-based similarity matching
- **Filtering System**: Genre, year, cast, director filters
- **Fuzzy Matching**: Name normalization and error tolerance
- **Ranking Algorithm**: Hybrid scoring with multiple factors

#### Data Layer
- **Vector Store**: FAISS index for semantic search
- **Movie Database**: Pandas DataFrame with movie metadata
- **Embeddings**: HuggingFace sentence transformers

## 3. Data Models

### 3.1 Core Data Structures

#### MovieChatState
```python
@dataclass
class MovieChatState:
    messages: List[Dict[str, str]]           # Chat history
    user_preferences: Dict[str, Any]         # Extracted preferences
    recommended_movies: List[Dict[str, Any]] # Current recommendations
    conversation_stage: str                  # Current dialog stage
    last_action: str                        # Last performed action
```

#### User Preferences Schema
```json
{
  "genres": ["action", "comedy"],
  "actors": ["Tom Hanks", "Meryl Streep"],
  "directors": ["Christopher Nolan"],
  "release_year": ["2020s", "1990s"],
  "query": "space adventure with robots"
}
```

#### Movie Record Schema
```python
{
    "id": int,
    "title": str,
    "year": int,
    "genres_cleaned": List[str],
    "cast": List[str],
    "directors": List[str],
    "keywords": List[str],
    "imdb_rating": float,
    "overview": str
}
```

### 3.2 Configuration Models

#### RecommendationConfig
```python
@dataclass
class RecommendationConfig:
    initial_top_k: int = 20
    final_top_k: int = 5
    fuzzy_match_threshold: float = 0.8
    year_range_tolerance: int = 2
    enable_hybrid_scoring: bool = False
    semantic_weight: float = 0.7
    popularity_weight: float = 0.2
    recency_weight: float = 0.1
```

## 4. Core Components

### 4.1 Recommendation Engine

#### Semantic Search Pipeline
1. **Query Processing**: Clean and normalize user query
2. **Vector Similarity**: FAISS similarity search against movie embeddings
3. **Initial Ranking**: Retrieve top-k candidates
4. **Filtering**: Apply genre, year, cast, director filters
5. **Re-ranking**: Hybrid scoring with semantic + popularity + recency
6. **Final Selection**: Return top recommendations

#### Filtering System
- **Genre Filter**: Exact and fuzzy matching against genre taxonomy
- **Year Filter**: Handles decades (90s), ranges (1990-2000), keywords (recent)
- **Person Filter**: Fuzzy matching for actors and directors with nickname support
- **Hybrid Scoring**: Configurable weights for different ranking factors

#### Fuzzy Matching Algorithm
```python
def fuzzy_name_search(query, name_list, threshold=50):
    # Normalize names and queries
    # Expand nicknames (Bob -> Robert)
    # Calculate similarity scores
    # Return top matches above threshold
```

### 4.2 Conversation Agent

#### Dialog Flow States
1. **Greeting**: Initial welcome and intent detection
2. **Gathering Preferences**: Extract user movie preferences
3. **Needs Preferences**: Request more specific information
4. **Preferences Gathered**: Sufficient info for recommendations
5. **Recommendations Ready**: Movies retrieved and ready
6. **Explained**: Recommendations with explanations added
7. **Refining**: Handle user feedback and adjustments

#### Preference Extraction
- **LLM-based**: Uses GPT-4 for structured preference extraction
- **JSON Schema**: Standardized output format
- **Merging Logic**: Combines new preferences with existing ones
- **Validation**: Ensures sufficient preferences for recommendations

#### Response Generation
- **Context-aware**: Considers conversation stage and history
- **Personalized**: Tailors responses to user preferences
- **Engaging**: Uses emojis and conversational tone
- **Error Handling**: Graceful degradation for failures

### 4.3 User Interface

#### Chat Interface Features
- **Real-time Updates**: Instant message display
- **Typing Indicators**: Simulated typing animation
- **Message History**: Persistent conversation log
- **Avatar Support**: User and bot profile images
- **Responsive Design**: Mobile-friendly layout

#### Styling System
- **Dark Theme**: Modern gradient backgrounds
- **Animations**: Smooth transitions and hover effects
- **Typography**: Clean, readable font hierarchy
- **Interactive Elements**: Hover states and button feedback

## 5. Technical Implementation

### 5.1 Technology Stack

#### Backend
- **Python 3.8+**: Core language
- **LangChain**: LLM integration and vector operations
- **FAISS**: Vector similarity search
- **Pandas**: Data manipulation
- **OpenAI GPT-4**: Natural language processing
- **HuggingFace**: Embedding models

#### Frontend
- **Gradio**: Web interface framework
- **HTML/CSS**: Custom styling
- **JavaScript**: Client-side interactions

#### Data Processing
- **Sentence Transformers**: Text embeddings
- **RapidFuzz**: Fuzzy string matching
- **JSON**: Structured data handling
- **CSV**: Movie database storage

### 5.2 Performance Optimizations

#### Caching Strategy
- **LRU Cache**: Fuzzy search results caching
- **Vector Index**: Pre-computed FAISS embeddings
- **Batch Processing**: Efficient data loading

#### Memory Management
- **Lazy Loading**: On-demand data loading
- **State Cleanup**: Automatic conversation state management
- **Resource Pooling**: Shared embedding models

### 5.3 Error Handling

#### Graceful Degradation
- **Fallback Responses**: Default recommendations on failures
- **Input Validation**: Sanitize user inputs
- **Exception Logging**: Comprehensive error tracking
- **User Feedback**: Friendly error messages

## 6. API Design

### 6.1 Core Methods

#### Recommendation Engine API
```python
def retrieve_semantic_recommendations(
    query: str,
    director: str = "",
    cast: str = "",
    genre: str = "All",
    year: str = "All",
    initial_top_k: int = 20,
    final_top_k: int = 5
) -> pd.DataFrame
```

#### Agent API
```python
def chat(
    message: str,
    state: Optional[MovieChatState] = None
) -> Tuple[str, MovieChatState]
```

#### Fuzzy Search API
```python
def fuzzy_name_search(
    query: str,
    name_list: List[str],
    threshold: int = 50,
    limit: int = 5
) -> List[Tuple[str, float]]
```

### 6.2 Configuration Management

#### Environment Variables
- `OPENAI_API_KEY`: OpenAI API authentication
- `DEVICE`: CUDA/CPU for embeddings
- `LOG_LEVEL`: Logging verbosity

#### Configuration Files
- `RecommendationConfig`: Tunable parameters
- `NICKNAME_MAP`: Name normalization mappings
- `GENRE_LIST`: Supported movie genres
- `YEAR_LIST`: Supported time periods

## 7. Security Considerations

### 7.1 Data Protection
- **API Key Management**: Secure credential storage
- **Input Sanitization**: Prevent injection attacks
- **Rate Limiting**: Prevent abuse of LLM calls
- **Session Management**: Secure state handling

### 7.2 Privacy
- **No Persistent Storage**: Conversations not saved
- **Local Processing**: Embeddings computed locally
- **Minimal Data**: Only necessary information retained

## 8. Testing Strategy

### 8.1 Unit Tests
- **Preference Extraction**: Validate JSON parsing
- **Fuzzy Matching**: Test name normalization
- **Filtering Logic**: Verify recommendation filtering
- **State Management**: Conversation flow testing

### 8.2 Integration Tests
- **End-to-end Workflows**: Complete recommendation flows
- **API Integration**: OpenAI and embedding services
- **Error Scenarios**: Failure mode testing

### 8.3 User Testing
- **Usability Testing**: Interface effectiveness
- **Recommendation Quality**: Accuracy assessment
- **Performance Testing**: Response time measurement

## 9. Deployment

### 9.1 Requirements
- **Python Dependencies**: Listed in requirements.txt
- **Model Files**: FAISS index and embeddings
- **Data Files**: Movie database CSV
- **Environment**: GPU-enabled instance recommended

### 9.2 Configuration
- **Port Settings**: Default Gradio port 7860
- **Resource Allocation**: Memory and GPU requirements
- **Logging Setup**: File and console output

### 9.3 Monitoring
- **Performance Metrics**: Response times and accuracy
- **Error Tracking**: Exception monitoring
- **Usage Analytics**: User interaction patterns

## 10. Future Enhancements

### 10.1 Planned Features (future)
- **Multi-language Support**: International movie databases
- **User Profiles**: Persistent preference learning
- **Social Features**: Movie sharing and ratings
- **Advanced Filtering**: More granular search options

### 10.2 Technical Improvements (future)
- **Streaming Responses**: Real-time response generation
- **Better Embeddings**: Domain-specific movie embeddings
- **Caching Layer**: Redis for production caching
- **API Endpoints**: RESTful API for external integration

### 10.3 UI/UX Enhancements (future)
- **Mobile App**: Native mobile interface
- **Voice Interface**: Speech-to-text integration
- **Rich Media**: Movie trailers and images
- **Accessibility**: Screen reader support

## 11. Conclusion

This movie recommendation chatbot represents a sophisticated blend of natural language processing, information retrieval, and user experience design. The modular architecture allows for easy maintenance and extension, while the modern web interface provides an engaging user experience.

The system successfully combines semantic search capabilities with traditional filtering methods, creating a robust recommendation engine that can handle diverse user queries and preferences. The conversational agent provides a natural interaction model that guides users through the recommendation process while maintaining context and personalization.

---

