import gradio as gr
import spaces
from typing import Dict, List, Any, Optional,Tuple, Union
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import json
import pandas as pd
import numpy as np
import torch
import os
import ast
import re

from dotenv import load_dotenv
load_dotenv()

client=OpenAI()

#data
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",    
                                   model_kwargs={'device': 'cuda'},    
                                   encode_kwargs={'normalize_embeddings': True})
vectorstore=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
movies=pd.read_csv('movies.csv')
json_columns=['keywords_cleaned',
              'production_countries_cleaned',
              'spoken_languages_cleaned',
              'cast',
              'directors',
              ]
for col in json_columns:
  movies[col]=movies[col].apply(json.loads)

director_list_list=movies['directors'].to_list()
director_list=[]
for i in director_list_list:
  for j in i:
    director_list.append(j)

cast_list_list=movies['cast'].to_list()
cast_list=[]
for i in cast_list_list:
  for j in i:
    cast_list.append(j)

movies.genres_cleaned=movies.genres_cleaned.apply(ast.literal_eval)
movies.genres_cleaned=movies.genres_cleaned.apply(lambda x: [i.lower() for i in x])




import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler('movie_chatbot.log'),
        logging.StreamHandler()  # This will still print to console
    ],
    force=True
)

logger = logging.getLogger(__name__)




from rapidfuzz import fuzz
# Nickname dictionary
NICKNAME_MAP = {
    "abby": ["abigail"], "ali": ["alison", "alice"], "ally": ["alison", "alice"],
    "andy": ["andrew"], "barb": ["barbara"], "beccy": ["rebecca"], "becky": ["rebecca"],
    "ben": ["benjamin"], "beth": ["elizabeth"], "bill": ["william"], "bob": ["robert"],
    "carrie": ["caroline", "carol"], "cathy": ["catherine", "katherine"],
    "charlie": ["charles", "charlotte"], "chuck": ["charles"], "danny": ["daniel"],
    "dan": ["daniel"], "dave": ["david"], "dick": ["richard"], "ed": ["edward", "edmund"],
    "eddie": ["edward", "edmund"], "frank": ["francis", "franklin"], "grace": ["gracie"],
    "hank": ["henry"], "harry": ["harold", "henry"], "jack": ["john"],
    "jackie": ["jacqueline"], "jake": ["jacob"], "jen": ["jennifer"],
    "jenny": ["jennifer"], "jerry": ["jerome", "gerald"], "jim": ["james"],
    "joey": ["joseph"], "joe": ["joseph"], "john": ["jonathan", "jon"],
    "jon": ["jonathan", "john"], "kate": ["katherine", "catherine"],
    "kathy": ["katherine", "catherine"], "larry": ["lawrence"], "leo": ["leonard", "leonardo"],
    "liz": ["elizabeth"], "luke": ["lucas", "lucius"], "maggie": ["margaret"],
    "mandy": ["amanda"], "marge": ["margaret"], "matt": ["matthew"], "meg": ["margaret"],
    "mike": ["michael"], "nancy": ["anne", "ann"], "nate": ["nathan", "nathaniel"],
    "nick": ["nicholas"], "pat": ["patrick", "patricia"], "peggy": ["margaret"],
    "pete": ["peter"], "rick": ["richard"], "rich": ["richard"], "rob": ["robert"],
    "ron": ["ronald"], "ronnie": ["ronald"], "sally": ["sarah"], "sam": ["samuel", "samantha"],
    "steve": ["steven", "stephen"], "sue": ["susan", "suzanne"], "suzie": ["susan", "suzanne"],
    "ted": ["edward", "theodore"], "tina": ["christina", "christine"], "tom": ["thomas"],
    "tony": ["anthony"], "trish": ["patricia"], "vicky": ["victoria"], "zack": ["zachary"],
}


def normalize(s):
    """Lowercase and remove non-alphanumeric characters"""
    return re.sub(r'\W+', '', s).lower()

def fuzzy_name_search(query, name_list, nickname_map=NICKNAME_MAP, threshold=50, limit=5):
    """Fuzzy search with nickname expansion and normalization.

    Args:
        query (str): Input name (possibly inaccurate).
        name_list (List[str]): List of full names to search.
        nickname_map (Dict[str, List[str]]): Optional nickname mappings.
        threshold (int): Minimum score to keep a match.
        limit (int): Max number of results to return.

    Returns:
        List[Tuple[str, float]]: Top matching names and scores.
    """
    # Normalize names and store mapping
    normalized_names = {normalize(name): name for name in name_list}
    normalized_query = normalize(query)

    # Expand query with nickname variants
    expanded_queries = {normalized_query}
    for nickname, full_forms in nickname_map.items():
        if nickname in normalized_query:
            for full in full_forms:
                variant = normalized_query.replace(nickname, normalize(full))
                expanded_queries.add(variant)

    # Fuzzy match each variant against the normalized names
    results = []
    for variant in expanded_queries:
        for norm_name, original_name in normalized_names.items():
            score = fuzz.ratio(variant, norm_name)
            if score >= threshold:
                results.append((original_name, score))

    # Deduplicate and sort by score
    results = sorted(set(results), key=lambda x: x[1], reverse=True)

    return results[:limit]




##########################################################################################################################################################################
##########################################################################################################################################################################
#Recommendation Engine
    

from dataclasses import dataclass
from functools import lru_cache

@dataclass
class RecommendationConfig:
    """Configuration for recommendation parameters"""
    initial_top_k: int = 20
    final_top_k: int = 5
    fuzzy_match_threshold: float = 0.8
    year_range_tolerance: int = 2
    enable_hybrid_scoring: bool = False
    semantic_weight: float = 0.7
    popularity_weight: float = 0.2
    recency_weight: float = 0.1

class ImprovedMovieRecommendationEngine:
    def __init__(self, movies_df: pd.DataFrame, vectorstore, director_list: List[str],
                 cast_list: List[str], genre_list: List[str], year_list: List[str],
                 config: Optional[RecommendationConfig] = None):
        """
        Initialize the recommendation engine with improved error handling and caching
        """
        self.movies = movies_df.copy()
        self.vectorstore = vectorstore 
        self.director_list = set(director_list)  # Use set for O(1) lookups
        self.cast_list = set(cast_list)
        self.genre_list = set(genre_list)  # Use set for O(1) lookups
        self.year_list = set(year_list)  # Use set for O(1) lookups
        self.config = config or RecommendationConfig()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Validate data
        self._validate_data()
    

    def _validate_data(self):
        """Validate the input data and log warnings for missing columns"""
        required_columns = ['id', 'title', 'year', 'genres_cleaned', 'directors', 'cast']
        missing_columns = [col for col in required_columns if col not in self.movies.columns]

        if missing_columns:
            self.logger.warning(f"Missing columns in movies DataFrame: {missing_columns}")

        # Check for common data issues
        if self.movies.empty:
            raise ValueError("Movies DataFrame is empty")

        self.logger.info(f"Initialized with {len(self.movies)} movies")

    @lru_cache(maxsize=1000)
    def _fuzzy_name_search_cached(self, name: str, name_type: str) -> Tuple[str, ...]:
        """Cached fuzzy name search to avoid repeated computations"""
        if name_type == 'director':
            name_list = self.director_list
        elif name_type == 'cast':
            name_list = self.cast_list
        elif name_type == 'genre':
            name_list = self.genre_list
        else:  # year
            name_list = self.year_list

        try:
            matches = fuzzy_name_search(name, list(name_list))
            return tuple(entry[0] for entry in matches[:3])  # Return top 3 matches
        except Exception as e:
            self.logger.error(f"Error in fuzzy search for {name}: {e}")
            return tuple()

    def _apply_genre_filter(self, df: pd.DataFrame, genre: str) -> pd.DataFrame:
        """Apply genre filter with improved performance and fuzzy matching"""
        if genre == "all":
            return df

        genre = genre.strip()

        # Try exact match first
        if genre in self.genre_list:
            filtered_df = df[df["genres_cleaned"].apply(
                lambda x: genre in x
            )]
            self.logger.info(f"Exact genre match for '{genre}': {len(filtered_df)} movies")
            return filtered_df

        # Fuzzy matching
        try:
            genre_guesses = self._fuzzy_name_search_cached(genre, "genre")

            if genre_guesses:
                filtered_df = df[df["genres_cleaned"].apply(
                    lambda x: any(g in genre_guesses for g in x)
                )]
                self.logger.info(f"Fuzzy genre match for '{genre}' -> {genre_guesses}: {len(filtered_df)} movies")
                return filtered_df
            else:
                self.logger.warning(f"No genre matches found for '{genre}'")
                return df

        except Exception as e:
            self.logger.warning(f"Error applying genre filter for '{genre}': {e}")
            return df

    def _apply_year_filter(self, df: pd.DataFrame, year: str) -> pd.DataFrame:
        """Apply year filter with improved range handling"""
        if year == "all" or year=='All':
            return df

        year = year.strip().lower()

        current_year=2025

        if year in ['recent', 'new', 'latest', 'modern']:
        # Recent movies (last 5 years)
          start_year = current_year - 5
          end_year = current_year+1
          filtered_df = df[df["year"].between(start_year, end_year+1)]
          self.logger.info(f"Recent movies filter: {len(filtered_df)} movies from {start_year}-{end_year}")
          return filtered_df

        if year in ['old', 'classic', 'vintage', 'retro']:
        # Old movies (before 1990)
          end_year = 1989
          filtered_df = df[df["year"] <= end_year]
          self.logger.info(f"Old movies filter: {len(filtered_df)} movies before {end_year + 1}")
          return filtered_df

        # Handle short decade formats like '90s', '80s', '00s'
        decade_match = re.match(r"(\d{1,2})s?$", year)
        if decade_match:
            decade_digits = decade_match.group(1)

            if len(decade_digits) == 1:
                # Single digit like '9' -> '2000s'
                if int(decade_digits) <= 2:
                    full_decade = f"200{decade_digits}s"
                else:
                    full_decade = f"199{decade_digits}s"
            elif len(decade_digits) == 2:
                # Two digits like '90' -> '1990s'
                if int(decade_digits) <= 30:  # Assume 00-30 means 2000s-2030s
                    full_decade = f"20{decade_digits}s"
                else:  # 31-99 means 1931-1999
                    full_decade = f"19{decade_digits}s"

            # Check if this converted decade exists in year_list
            if full_decade in [y.lower() for y in self.year_list]:
                decade_start = int(full_decade[:4])
                start_year = decade_start
                end_year = decade_start + 9
                filtered_df = df[df["year"].between(start_year, end_year)]
                self.logger.info(f"Short decade format '{year}' -> {full_decade}: {len(filtered_df)} movies")
                return filtered_df

        # Try exact match first against year_list (e.g., "1990s")
        if year in year_list:
            # Handle decade specification
            if year.endswith('s'):
                decade_start = int(year[:4])
                start_year = decade_start
                end_year = decade_start + 9
                filtered_df = df[df["year"].between(start_year, end_year)]
                self.logger.info(f"Exact decade match for '{year}': {len(filtered_df)} movies")
                return filtered_df

        # Try fuzzy matching against year_list
        try:
            year_guesses = self._fuzzy_name_search_cached(year, "year")

            if year_guesses:
                # Use the first fuzzy match
                matched_year = year_guesses[0]
                if matched_year.endswith('s'):
                    decade_start = int(matched_year[:4])
                    start_year = decade_start
                    end_year = decade_start + 9
                    filtered_df = df[df["year"].between(start_year, end_year)]
                    self.logger.info(f"Fuzzy year match for '{year}' -> {matched_year}: {len(filtered_df)} movies")
                    return filtered_df
        except Exception as e:
            self.logger.warning(f"Error in fuzzy year matching: {e}")

        # Fallback to original logic for specific years
        try:
            # Extract year from string (handles formats like "2020", "2020-2025")
            year_match = re.search(r'(\d{4})', str(year))
            if not year_match:
                self.logger.warning(f"Could not extract year from '{year}'")
                return df

            target_year = int(year_match.group(1))
            tolerance = self.config.year_range_tolerance

            # Single year with tolerance
            start_year = target_year - tolerance
            end_year = target_year + tolerance

            filtered_df = df[df["year"].between(start_year, end_year)]
            self.logger.info(f"Specific year match for '{year}': {len(filtered_df)} movies")
            return filtered_df

        except Exception as e:
            self.logger.error(f"Error applying year filter for '{year}': {e}")
            return df

    def _apply_person_filter(self, df: pd.DataFrame, person: str, person_type: str) -> pd.DataFrame:
        """Apply director or cast filter with improved fuzzy matching"""
        if not person.strip():
            return df

        person = person.strip()
        column_name = "directors" if person_type == "director" else "cast"
        person_list = self.director_list if person_type == "director" else self.cast_list

        # Try exact match first
        if person in person_list:
            filtered_df = df[df[column_name].apply(lambda x: isinstance(x, list) and person in x)]
            self.logger.info(f"Exact {person_type} match for '{person}': {len(filtered_df)} movies")
            return filtered_df

        # Fuzzy matching
        try:
            person_guesses = self._fuzzy_name_search_cached(person, person_type)

            if person_guesses:
                filtered_df = df[df[column_name].apply(
                    lambda x: isinstance(x, list) and any(p in person_guesses for p in x)
                )]
                self.logger.info(f"Fuzzy {person_type} match for '{person}' -> {person_guesses}: {len(filtered_df)} movies")
                return filtered_df
            else:
                self.logger.warning(f"No {person_type} matches found for '{person}'")
                return df

        except Exception as e:
            self.logger.error(f"Error in {person_type} search for '{person}': {e}")
            return df
            

    def _apply_semantic_search(self, df: pd.DataFrame, query: str, initial_top_k: int) -> pd.DataFrame:
        """Apply semantic search with improved error handling"""
        if not query.strip():
            return df.head(initial_top_k)

        try:
            # Get semantic recommendations
            semantic_results = self.vectorstore.similarity_search(query, k=initial_top_k)

            if not semantic_results:
                self.logger.warning(f"No semantic results found for query: '{query}'")
                return df.head(initial_top_k)

            # Extract movie IDs from results
            movie_ids = []
            for rec in semantic_results:
                try:
                    # Handle different content formats
                    content = rec.page_content.strip('"').strip()
                    movie_id = int(content.split()[0])
                    movie_ids.append(movie_id)
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error parsing semantic result: {content}, error: {e}")
                    continue

            if not movie_ids:
                self.logger.warning("No valid movie IDs extracted from semantic search")
                return df.head(initial_top_k)

            # Filter and preserve semantic ranking order
            semantic_df = df[df["id"].isin(movie_ids)]

            # Reorder based on semantic ranking
            semantic_df = semantic_df.set_index('id').reindex(movie_ids).reset_index()
            semantic_df = semantic_df.dropna(subset=['title'])  # Remove any failed matches

            self.logger.info(f"Semantic search for '{query}': {len(semantic_df)} movies")
            return semantic_df.head(initial_top_k)

        except Exception as e:
            self.logger.error(f"Error in semantic search for '{query}': {e}")
            return df.head(initial_top_k)

    def _calculate_hybrid_score(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Calculate hybrid scores combining semantic similarity, popularity, and recency"""
        if not self.config.enable_hybrid_scoring or df.empty:
            return df

        try:
            # Normalize scores to 0-1 range
            df = df.copy()

            # Semantic score (based on ranking position)
            df['semantic_score'] = np.linspace(1, 0, len(df))

            # Popularity score (assuming higher ratings/votes indicate popularity)
            if 'imdb_rating' in df.columns:
                df['popularity_score'] = (df['imdb_rating'] - df['imdb_rating'].min()) / (df['imdb_rating'].max() - df['imdb_rating'].min())
            else:
                df['popularity_score'] = 0.5  # Default neutral score

            # Recency score (more recent movies get higher scores)
            if 'year' in df.columns:
                current_year = pd.Timestamp.now().year
                df['recency_score'] = np.clip((df['year'] - 1900) / (current_year - 1900), 0, 1)
            else:
                df['recency_score'] = 0.5  # Default neutral score

            # Calculate hybrid score
            df['hybrid_score'] = (
                self.config.semantic_weight * df['semantic_score'] +
                self.config.popularity_weight * df['popularity_score'] +
                self.config.recency_weight * df['recency_score']
            )

            # Sort by hybrid score
            df = df.sort_values('hybrid_score', ascending=False)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating hybrid scores: {e}")
            return df

    def retrieve_semantic_recommendations(
        self,
        query: str,
        director: str = "",
        cast: str = "",
        genre: str = "All",
        year: str = "All",
        initial_top_k: int = 20,
        final_top_k: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Enhanced movie recommendation retrieval with improved filtering and error handling

        Args:
            query: User's description of the movie, OTHER THAN genre, cast, year, director
            director: Director name (fuzzy matching supported)
            cast: Cast member name (fuzzy matching supported)
            genre: Genre filter (fuzzy matching supported)
            year: Year filter (supports ranges and decades)

            initial_top_k: Initial number of movies to retrieve
            final_top_k: Final number of movies to return
            **kwargs: Additional parameters for future extensibility

        Returns:
            pd.DataFrame: Recommended movies with enhanced metadata
        """
        print('='*25)
        print(f'query: {query}, director: {director}, cast: {cast}, genre: {genre}, year: {year}')
        logger.info(f'Searching: query: {query}, director: {director}, cast: {cast}, genre: {genre}, year: {year}')
        print('='*25)
        # Use config defaults if not specified
        initial_top_k = self.config.initial_top_k
        final_top_k = self.config.final_top_k

        self.logger.info(f"Starting recommendation search with query: '{query}'")

        try:
            # Start with full dataset
            movie_rec = self.movies.copy()
            initial_count = len(movie_rec)

            # Apply hard filters first (most restrictive)
            movie_rec = self._apply_genre_filter(movie_rec, genre)
            movie_rec = self._apply_year_filter(movie_rec, year)
            print(f"Initial movie count after filters: {len(movie_rec)}")
            logger.info(f"Initial movie count after filters: {len(movie_rec)}")

            # Apply flexible filters
            movie_rec = self._apply_person_filter(movie_rec, director, "director")
            movie_rec = self._apply_person_filter(movie_rec, cast, "cast")
            print(f"Movie count after person filters: {len(movie_rec)}")
            logger.info(f"Movie count after person filters: {len(movie_rec)}")

            # Check if we have any movies left after filtering
            if movie_rec.empty:
                self.logger.warning("No movies found after applying filters, returning top movies")
                movie_rec = self.movies.copy()

            # Apply semantic search
            movie_rec = self._apply_semantic_search(movie_rec, query, initial_top_k)

            # Calculate hybrid scores if enabled
            movie_rec = self._calculate_hybrid_score(movie_rec, query)

            # Final selection
            final_recommendations = movie_rec.head(final_top_k)

            # Add recommendation metadata
            final_recommendations = final_recommendations.copy()
            final_recommendations['recommendation_score'] = np.linspace(1.0, 0.1, len(final_recommendations))
            final_recommendations['search_query'] = query
            final_recommendations['filters_applied'] = {
                'genre': genre,
                'year': year,
                'director': director,
                'cast': cast
            }

            self.logger.info(f"Recommendation complete: {initial_count} -> {len(movie_rec)} -> {len(final_recommendations)} movies")

            return final_recommendations

        except Exception as e:
            self.logger.error(f"Error in recommendation process: {e}")
            # Return fallback recommendations
            return self.movies.head(final_top_k)

    def get_recommendation_stats(self) -> Dict:
        """Get statistics about the recommendation engine"""
        return {
            'total_movies': len(self.movies),
            'total_directors': len(self.director_list),
            'total_cast': len(self.cast_list),
            'total_genres': len(self.genre_list),
            'total_years': len(self.year_list),
            'config': self.config.__dict__
        }

    def suggest_similar_names(self, name: str, name_type: str = 'director', top_k: int = 5) -> List[str]:
        """Suggest similar names for typos or partial matches"""
        try:
            matches = self._fuzzy_name_search_cached(name, name_type)
            return list(matches[:top_k])
        except Exception as e:
            self.logger.error(f"Error getting name suggestions: {e}")
            return []



def create_recommendation_engine(movies_df, vectorstore_loader_func, director_list, cast_list, genre_list, year_list,
                               config_overrides: Optional[Dict] = None):
    """Factory function to create a configured recommendation engine"""

    # Create custom configuration if needed
    config = RecommendationConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return ImprovedMovieRecommendationEngine(
        movies_df, vectorstore, director_list, cast_list, genre_list, year_list, config
    )

genre_list=['all','action','adventure','animation','comedy','crime','documentary','drama',
            'family','fantasy','history','horror','music','mystery','romance','science fiction',
            'tv movie','thriller','war','western']

year_list = ['all','2020s', '2010s', '2000s', '1990s', '1980s', '1970s', '1960s', '1950s', '1940s', '1930s', '1920s', '1910s']
  
# Example configuration for different use cases
FAST_CONFIG = {
    'initial_top_k': 30,
    'final_top_k': 10,
    'enable_hybrid_scoring': False
}

COMPREHENSIVE_CONFIG = {
    'initial_top_k': 100,
    'final_top_k': 25,
    'enable_hybrid_scoring': True,
    'semantic_weight': 0.6,
    'popularity_weight': 0.3,
    'recency_weight': 0.1
}


##########################################################################################################################################################################
##########################################################################################################################################################################
#Create Agent
from dataclasses import dataclass, field

@dataclass
class MovieChatState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    recommended_movies: List[Dict[str, Any]] = field(default_factory=list)
    conversation_stage: str = "greeting"
    last_action: str = ""

class MovieRecommendationAgent:
    def __init__(self, client, movie_db, recommendation_engine):
        self.client = client
        self.movie_db = movie_db
        self.recommendation_engine = recommendation_engine

        # Define the workflow as a simple mapping
        self.workflow_map = {
            "greeting": self._handle_greeting,
            "gathering_preferences": self._handle_preference_gathering,
            "needs_preferences": self._handle_preference_gathering,
            "preferences_gathered": self._handle_recommendations,
            "recommendations_ready": self._handle_explanation,
            "explained": self._handle_response,
            "refining": self._handle_refinement,
        }

    def _call_llm(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content    

    def chat(self, message: str, state: Optional[MovieChatState] = None) -> Tuple[str, MovieChatState]:
        """Main chat interface - simplified version of the graph execution"""
        if state is None:
            state = MovieChatState()
        logger.info('chatting')
        # Add user message
        state.messages.append({"role": "user", "content": message})

        # Process through the workflow
        response = self._process_conversation(state)

        # Add AI response
        state.messages.append({"role": "assistant", "content": response})

        return response, state

    def _process_conversation(self, state: MovieChatState) -> str:
        """Process the conversation through multiple stages until we have a response"""
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get the handler for current stage
            handler = self.workflow_map.get(state.conversation_stage)
            if not handler:
                # Default to response generation if stage is unknown
                return self._generate_response(state)

            # Execute the handler
            next_stage = handler(state)

            # If we get a response, return it
            if next_stage == "respond":
                return self._generate_response(state)

            # Update stage and continue
            if next_stage:
                state.conversation_stage = next_stage
            else:
                # If no next stage specified, generate response
                return self._generate_response(state)

        # Fallback if we hit max iterations
        return self._generate_response(state)

    def _handle_greeting(self, state: MovieChatState) -> str:
        """Handle initial greeting and route conversation"""
        last_message = state.messages[-1]["content"] if state.messages else ""

        # Check if user is asking for recommendations directly
        recommendation_keywords = ['recommend', 'suggest', 'movie', 'watch', 'film','something','looking']
        if any(word in last_message.lower() for word in recommendation_keywords):
            if not state.user_preferences:
                state.conversation_stage = "needs_preferences"
                state.last_action = "needs_preferences"
                return "gathering_preferences"
            else:
                state.conversation_stage = "preferences_gathered"
                state.last_action = "has_preferences"
                return "preferences_gathered"
        else:
            state.conversation_stage = "greeting"
            state.last_action = "greeted"
            return "respond"

    def _handle_preference_gathering(self, state: MovieChatState) -> str:
        """Extract and store user preferences"""
        last_message = state.messages[-1]["content"]

        # Extract preferences using LLM
        new_prefs = self._extract_preferences(last_message, state.user_preferences)

        # Merge preferences
        self._merge_preferences(state.user_preferences, new_prefs)

        state.last_action = "gathered_preferences"

        # Check if we have enough preferences
        if self._has_sufficient_preferences(state.user_preferences):
            state.conversation_stage = "preferences_gathered"
            return "preferences_gathered"
        else:
            state.conversation_stage = "needs_preferences"
            return "respond"  # Ask for more preferences

    def _handle_recommendations(self, state: MovieChatState) -> str:
        """Generate recommendations"""
        try:
            recommendations = self._generate_recommendations(state.user_preferences)
            state.recommended_movies = recommendations
            state.conversation_stage = "recommendations_ready"
            state.last_action = "generated_recommendations"
            return "recommendations_ready"
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            state.conversation_stage = "error"
            return "respond"

    def _handle_explanation(self, state: MovieChatState) -> str:
        """Add explanations to recommendations"""
        for movie in state.recommended_movies:
            movie['explanation'] = self._create_explanation(movie, state.user_preferences)

        state.conversation_stage = "explained"
        state.last_action = "explained_recommendations"
        return "explained"

    def _handle_refinement(self, state: MovieChatState) -> str:
        """Handle user feedback and refinement"""
        last_message = state.messages[-1]["content"]

        # Extract feedback and adjust preferences
        adjustments = self._extract_feedback(last_message, state.recommended_movies)
        self._apply_preference_adjustments(state.user_preferences, adjustments)

        state.conversation_stage = "preferences_gathered"
        state.last_action = "refined_preferences"
        return "preferences_gathered"

    def _handle_response(self, state: MovieChatState) -> str:
        """Signal that we should generate a response"""
        return "respond"


    def _extract_preferences(self, message: str, current_prefs: Dict) -> Dict:
        """Extract preferences from user message using LLM with simplified structure"""
        prompt = f"""You are a movie preference analyzer. Your task is to extract specific movie-related information from user input and organize it into primary fields and a specific query.

## Primary Extraction Fields:

### 1. Genres
- Extract any mentioned movie genres (action, comedy, drama, horror, sci-fi, romance, thriller, documentary, etc.)
- Include both explicit mentions ("I love action movies") and implicit ones ("Marvel films are great")

### 2. Actors
- Extract names of actors mentioned
- Note whether they're mentioned positively or negatively

### 3. Directors
- Extract names of directors mentioned
- Note whether they're mentioned positively or negatively

### 4. Release Year
- Extract specific years, decades, or time periods mentioned
- Include ranges (e.g., "2010-2020", "90s", "recent", "classic")

### 5. Query
- Keywords or phrases from user input that doesn't fit into the above categories.
- This includes: mood preferences, themes, specific movie titles, viewing context, format preferences, plot elements, etc.
- ONLY include specific movie-related information, DO NOT USE generic query. For example, when user say 'looking for a good movie' or 'recommend some movies', the query should be empty.

## Instructions:
1. Extract only the four primary fields (genres, actors, directors, release_year)
2. If the message provides any specific movie-related information, include it in the query field. Otherwise, leave it empty.
3. For negative mentions, prefix with "NOT" (e.g., "NOT horror" for genres)
4. Use consistent formatting for names and titles

Current preferences: {json.dumps(current_prefs)}

User message: "{message}"

Return only valid JSON in this exact format:
{{
  "genres": [],
  "actors": [],
  "directors": [],
  "release_year": [],
  "query": ""
}}"""

        try:
            response = self._call_llm(prompt)
            response_text = response.strip()

            # Debug: print the actual response
            print(f"LLM Response: '{response_text}'")
            logger.info(f"LLM Response: '{response_text}'")

            # Try to extract JSON if it's wrapped in markdown or other text
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()

            # Remove any leading/trailing whitespace and newlines
            response_text = response_text.strip()

            if not response_text:
                raise ValueError("Empty response from LLM")

            parsed_response = json.loads(response_text)

            # Validate the structure
            expected_keys = {"genres", "actors", "directors", "release_year", "query"}
            if not all(key in parsed_response for key in expected_keys):
                raise ValueError(f"Missing required keys. Got: {list(parsed_response.keys())}")

            return parsed_response

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error parsing LLM response: {e}")
            logger.error(f"Error parsing LLM response: {e}")
            print(f"Raw response: {getattr(response, 'content', 'No content')}")
            logger.info(f"Raw response: {getattr(response, 'content', 'No content')}")
            # Fallback: add message to query
            return {
                "genres": [],
                "actors": [],
                "directors": [],
                "release_year": [],
                "query": message
            }

    def _merge_preferences(self, current_prefs: Dict, new_prefs: Dict) -> None:
        """Merge new preferences into current preferences"""
        for key, value in new_prefs.items():
            if key == 'query':
                # Combine query strings
                if 'query' not in current_prefs:
                    current_prefs['query'] = ""
                if current_prefs['query'] and value:
                    current_prefs['query'] += " " + value
                elif value:
                    current_prefs['query'] = value
            elif isinstance(value, list) and key in current_prefs:
                # Extend lists and remove duplicates
                if isinstance(current_prefs[key], list):
                    current_prefs[key].extend(value)
                    current_prefs[key] = list(set(current_prefs[key]))  # Remove duplicates
                else:
                    current_prefs[key] = [current_prefs[key]] + value
            elif value:  # Only add non-empty values
                current_prefs[key] = value

    def _has_sufficient_preferences(self, prefs: Dict) -> bool:
        """Check if we have enough preferences to make recommendations"""
        has_query = bool(prefs.get('query', '').strip())
        has_specific_criteria = any(prefs.get(key) for key in ['genres', 'actors', 'directors', 'release_year'])
        return has_query or has_specific_criteria

    def _generate_recommendations(self, preferences: Dict) -> List[Dict]:
        """Generate movie recommendations using the recommendation engine"""
        # Map preferences to function parameters
        query = preferences.get("query", "")
        director = preferences.get("directors", [""])[0] if preferences.get("directors") else ""
        cast = preferences.get("actors", [""])[0] if preferences.get("actors") else ""
        genre = preferences.get("genres", ["all"])[0] if preferences.get("genres") else "all"

        # Handle release_year - could be year, decade, or range
        year_pref = preferences.get("release_year")
        if year_pref and isinstance(year_pref, list) and year_pref:
            year = str(year_pref[0])
        else:
            year = "all"

        # Use recommendation engine
        recommendations_df = self.recommendation_engine.retrieve_semantic_recommendations(
            query=query,
            director=director,
            cast=cast,
            genre=genre,
            year=year,
            final_top_k=5
        )

        return recommendations_df.to_dict('records') if not recommendations_df.empty else []

    def _create_explanation(self, movie: Dict, preferences: Dict) -> str:
        """Create explanation for why a movie was recommended"""
        explanations = []

        # Check for genre matches
        if 'genres' in preferences:
            movie_genres = movie.get('genres_cleaned', movie.get('genres', []))
            if isinstance(movie_genres, str):
                movie_genres = [movie_genres]
            user_genres = preferences['genres'] if isinstance(preferences['genres'], list) else [preferences['genres']]

            if any(genre.lower() in str(movie_genres).lower() for genre in user_genres):
                explanations.append(f"matches your interest in {', '.join(user_genres)}")

        # Check for actor matches
        if 'actors' in preferences and 'cast' in movie:
            explanations.append(f"features your preferred actors")

        # Check for director matches
        if 'directors' in preferences and 'directors' in movie:
            explanations.append(f"directed by your preferred director")

        # Check for year matches
        if 'release_year' in preferences:
            explanations.append(f"from your preferred time period")

        # Generic explanation if nothing specific
        if not explanations and preferences.get('query'):
            explanations.append("matches your description")

        return "; ".join(explanations) if explanations else "recommended based on your preferences"

    def _extract_feedback(self, feedback: str, movies: List[Dict]) -> Dict:
        """Extract user feedback about recommendations"""
        # Simple keyword-based feedback extraction
        # You could replace this with LLM-based extraction
        adjustments = {}

        feedback_lower = feedback.lower()

        if any(word in feedback_lower for word in ['not', 'don\'t', 'dislike', 'hate']):
            # Negative feedback - could extract what to avoid
            adjustments['negative_feedback'] = True

        if any(word in feedback_lower for word in ['more', 'similar', 'like this']):
            # Positive feedback - could extract what to emphasize
            adjustments['positive_feedback'] = True

        return adjustments

    def _apply_preference_adjustments(self, preferences: Dict, adjustments: Dict) -> None:
        """Apply refinement adjustments to preferences"""
        # Simple implementation - you can make this more sophisticated
        if adjustments.get('negative_feedback'):
            # Could add exclusion preferences
            pass

        if adjustments.get('positive_feedback'):
            # Could emphasize current preferences
            pass

    def _generate_response(self, state: MovieChatState) -> str:
        """Generate the final response to the user"""
        prompt = f"""
        You are a friendly movie recommendation chatbot with a light sense of humor.

        Conversation stage: {state.conversation_stage}
        Last action: {state.last_action}
        User preferences: {json.dumps(state.user_preferences)}

        Chat history (last 3 messages):
        {self._format_recent_messages(state.messages)}

        Recommended movies: {json.dumps(state.recommended_movies[:5] if state.recommended_movies else [])}

        Generate an appropriate response based on the current stage:
        - If greeting: Welcome the user and ask how you can help, DO NOT recommend movies at this stage
        - If needs_preferences: Ask for their movie preferences naturally
        - If explained: Present the recommendations engagingly with brief explanations
        - If error: Apologize and ask them to try rephrasing their request

        Keep responses conversational, helpful, and concise. Use emojis where appropriate.

        """

        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            return "I'm having trouble processing your request right now. Could you try rephrasing what you're looking for?"

    def _format_recent_messages(self, messages: List[Dict], n: int = 3) -> str:
        """Format recent messages for context"""
        recent = messages[-n:] if len(messages) > n else messages
        formatted = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)


import gradio as gr
from datetime import datetime
import threading
import time

rec_engine=ImprovedMovieRecommendationEngine(movies, vectorstore, director_list, cast_list,genre_list,year_list)
movie_agent = MovieRecommendationAgent(
    client=client,
    movie_db=movies, # This is the full movie DataFrame
    recommendation_engine=rec_engine
)

import gradio as gr
from datetime import datetime
import threading
import time

# Your existing code (rec_engine, movie_agent setup)
rec_engine = ImprovedMovieRecommendationEngine(movies, vectorstore, director_list, cast_list, genre_list, year_list)
movie_agent = MovieRecommendationAgent(
    client=client,
    movie_db=movies,
    recommendation_engine=rec_engine
)

# Enhanced chatbot function with typing indicator
def chatbot_response_with_typing(message: str, history: list, chat_state):
    """Enhanced chatbot response with typing simulation"""
    if not message.strip():
        return "", history, chat_state
    
    # Add user message immediately
    history.append([message, None])
    
    # Simulate typing with dots
    for i in range(3):
        typing_msg = "●" * (i + 1) + "○" * (2 - i)
        history[-1][1] = typing_msg
        yield "", history, chat_state
        time.sleep(0.3)
    
    # Get actual response
    response_content, updated_chat_state = movie_agent.chat(message, chat_state)
    
    # Update with final response
    history[-1][1] = response_content
    
    return "", history, updated_chat_state


# Custom CSS for modern dark styling
custom_css = """
/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #ffffff;
}

/* Header styling */
.header-container {
    background: rgba(30, 30, 50, 0.8);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
    text-align: center;
    border: 1px solid rgba(100, 100, 150, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.header-title {
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(45deg, #64b5f6, #42a5f5, #2196f3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.header-subtitle {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.9);
    margin-bottom: 20px;
}

/* Chat interface styling */
.chat-container {
    background: rgba(20, 20, 35, 0.9);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(100, 100, 150, 0.2);
    backdrop-filter: blur(10px);
}

/* Message styling */
.message {
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    max-width: 80%;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease;
}

.message:hover {
    transform: translateY(-2px);
}

.user-message {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 5px;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.bot-message {
    background: linear-gradient(135deg, #374151, #4b5563);
    color: #e5e7eb;
    margin-right: auto;
    border-bottom-left-radius: 5px;
    border: 1px solid rgba(75, 85, 99, 0.3);
}

/* Input area styling */
.input-container {
    background: rgba(30, 30, 50, 0.8);
    border-radius: 25px;
    padding: 15px;
    margin-top: 15px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    border: 2px solid rgba(100, 100, 150, 0.3);
    transition: border-color 0.3s ease;
}

.input-container:focus-within {
    border-color: #3b82f6;
}

/* Button styling */
.custom-button {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 12px 25px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
    background: linear-gradient(135deg, #2563eb, #60a5fa);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(30, 30, 50, 0.5);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Responsive design */
@media (max-width: 768px) {
    .header-title {
        font-size: 2rem;
    }
    
    .header-subtitle {
        font-size: 1rem;
    }
    
    .message {
        max-width: 95%;
        padding: 12px;
    }
}

/* Chatbot specific styling */
.chatbot {
    border: none !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
    background: rgba(20, 20, 35, 0.9) !important;
}

/* Message input styling */
.message-input {
    border-radius: 20px !important;
    border: 2px solid rgba(100, 100, 150, 0.3) !important;
    background: rgba(30, 30, 50, 0.8) !important;
    color: #ffffff !important;
    font-size: 16px !important;
    transition: border-color 0.3s ease !important;
}

.message-input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

.message-input::placeholder {
    color: rgba(255, 255, 255, 0.6) !important;
}

/* Dark theme for gradio components */
.gradio-container .gr-button {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
    color: white !important;
    border: none !important;
}

.gradio-container .gr-button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
}

/* Sidebar styling */
.sidebar-content {
    background: rgba(30, 30, 50, 0.8);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(100, 100, 150, 0.3);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
}

.sidebar-content h3 {
    color: #64b5f6;
    margin-bottom: 15px;
}

.sidebar-content div {
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Examples styling */
.examples-container {
    background: rgba(30, 30, 50, 0.6);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(100, 100, 150, 0.2);
}

.examples-container .gr-button {
    background: rgba(30, 30, 50, 0.8) !important;
    color: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(100, 100, 150, 0.3) !important;
    margin: 5px !important;
}

.examples-container .gr-button:hover {
    background: rgba(59, 130, 246, 0.3) !important;
    border-color: #3b82f6 !important;
}
"""


# Create the Gradio interface
with gr.Blocks(css=custom_css, title="🎬 Movie Recommendation Bot") as demo:
    # Header section
    with gr.Row():
        gr.HTML("""
        <div class="header-container fade-in">
            <div class="header-title">🎬 ReelTalk</div>
            <div class="header-subtitle">Your Personal Movie Recommendation Expert</div>
            <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">
                Powered by AI • Discover your next favorite film
            </div>
        </div>
        """)
    
    # Main chat interface
    with gr.Row():
        with gr.Column(scale=1):
            # Chat history
            chatbot = gr.Chatbot(
                height=600,
                label="",
                show_label=False,
                container=True,
                elem_classes=["chat-container"], 
                avatar_images=("https://cdn-icons-png.flaticon.com/512/149/149071.png",
               "https://cdn-icons-png.flaticon.com/512/2503/2503508.png"),
            )
            
            # Input area
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="🍿 Ask me about films!",
                    container=False,
                    scale=4,
                    elem_classes=["message-input"], 
                )
                
                with gr.Column(scale=1, min_width=100):
                    submit_btn = gr.Button(
                        "Send 🚀",
                        variant="primary",
                        elem_classes=["custom-button"]
                    )
            
            # Action buttons
            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️ Clear Chat",
                    variant="secondary",
                    elem_classes=["custom-button"]
                )
                
                example_btn = gr.Button(
                    "💡 Show Examples",
                    variant="secondary",
                    elem_classes=["custom-button"]
                )
    
    # Sidebar with quick actions
    with gr.Column(scale=0.3, min_width=200):
        gr.HTML("""
        <div style="background: rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 20px; margin-bottom: 20px;">
            <h3 style="color: white; margin-bottom: 15px;">🎯 Quick Actions</h3>
            <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; line-height: 1.6;">
                • Ask for recommendations by genre<br>
                • Search by actor or director<br>
                • Find movies by year or decade<br>
                • Get similar movie suggestions<br>
                • Ask about movie plots or details
            </div>
        </div>
        """)
        
        # Example queries
        examples = [
            "Recommend me a good action movie",
            "I want to watch something like The Godfather",
            "Can you suggest a comedy from the 1990s?",
            "Show me movies directed by Christopher Nolan"
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[msg],
            label="💭 Try these examples:"
        )
    
    # State management
    chat_state_store = gr.State(MovieChatState())
    
    # Enhanced user message handler
    def user_message_handler(user_message, history, chat_state):
        """Enhanced message handler with better error handling"""
        try:
            if not user_message.strip():
                return "", history, chat_state
                
            # Add timestamp to user message
            timestamp = datetime.now().strftime("%I:%M %p")
            
            response_content, updated_chat_state = movie_agent.chat(user_message, chat_state)
            
            # Format response with emoji and styling
            formatted_response = f"🎬 {response_content}"
            
            # Update history
            history.append([user_message, formatted_response])
            
            return "", history, updated_chat_state
            
        except Exception as e:
            error_msg = f"🚨 I encountered an error: {str(e)}. Please try again!"
            history.append([user_message, error_msg])
            return "", history, chat_state
    
    # Show examples function
    def show_examples():
        example_text = """Here are some example queries you can try:
        
🎭 **Genre-based**: "Recommend some action movies from the 90s"
🎬 **Actor-based**: "What are the best Tom Hanks movies?"
🎪 **Director-based**: "Show me Christopher Nolan films"
🎨 **Mood-based**: "I want something funny and light-hearted"
🎯 **Keyword-based**: "Looking for a romantic vampire movie"
📅 **Era-based**: "Classic movies from the 1980s"
        """
        return example_text
    
    # Event handlers
    msg.submit(
        user_message_handler,
        inputs=[msg, chatbot, chat_state_store],
        outputs=[msg, chatbot, chat_state_store]
    )
    
    submit_btn.click(
        user_message_handler,
        inputs=[msg, chatbot, chat_state_store],
        outputs=[msg, chatbot, chat_state_store]
    )
    
    clear_btn.click(
        lambda: (None, [], MovieChatState()),
        outputs=[msg, chatbot, chat_state_store]
    )
    
    example_btn.click(
        lambda: show_examples(),
        outputs=[msg]
    )
    
    # Welcome message
    demo.load(
        lambda: [["👋 Hi", "🎬 Hello!I'm your personal movie recommendation assistant. I can help you discover amazing films based on your preferences. What kind of movie are you in the mood for today?"]],
        outputs=[chatbot]
    )


if __name__ == "__main__":
    demo.launch()
