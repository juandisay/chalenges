import os
import json
from typing import List, Dict, Any
from difflib import SequenceMatcher

class DataSearch:
    def __init__(self, cities: List[Dict[str, Any]]):
        """
        Initialize with list of cities
        """
        self.cities = cities

    def _similarity_ratio(self, search_term: str, target: str) -> float:
        """
        Calculate similarity ratio between search term and target string
        using multiple matching strategies
        """
        search_term = search_term.lower()
        target = target.lower()

        # Strategy 1: Check if search term is a substring of target
        if search_term in target:
            return 1.0

        # Strategy 2: Check individual words matching
        search_words = set(search_term.split())
        target_words = set(target.split())
        
        # If all search words are found in target words
        if search_words.issubset(target_words):
            return 1.0

        # Strategy 3: Use SequenceMatcher for fuzzy matching
        # Try matching full strings
        full_ratio = SequenceMatcher(None, search_term, target).ratio()
        
        # Try matching with each word in the target
        target_parts = target.split()
        word_ratios = [
            SequenceMatcher(None, search_term, part).ratio()
            for part in target_parts
        ]
        best_word_ratio = max(word_ratios) if word_ratios else 0

        # Return the best match ratio
        return max(full_ratio, best_word_ratio)

    def search(self, search_term: str, min_similarity: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for cities using enhanced string matching
        
        Args:
            search_term: Name of city to search for
            min_similarity: Minimum similarity ratio (0 to 1) for matching
            
        Returns:
            List of matching city dictionaries
        """
        matches = []
        for city in self.cities:
            similarity = self._similarity_ratio(search_term, city['name'])
            if similarity >= min_similarity:
                matches.append({
                    **city,
                    '_match_score': similarity  # Add match score for debugging
                })
        
        # Sort by similarity in descending order
        matches.sort(key=lambda x: x['_match_score'], reverse=True)
        
        # Remove _match_score before returning
        for match in matches:
            match.pop('_match_score', None)
            
        return matches


def search(value: str) -> List[Dict[str, Any]]:
    """
    Search for cities in the JSON file
    
    Args:
        value: Search term
        
    Returns:
        List of matching cities
    """
    # Get the directory containing your script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'list_country.json')

    # Now use file_path to open your file
    with open(file_path, 'r') as f:
        data = json.load(f)

    initialize = DataSearch(data)
    # Using a lower min_similarity to catch more potential matches
    return initialize.search(value, min_similarity=0.8)
