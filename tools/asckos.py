import requests
import pandas as pd
import os
import re

class ReactionClassifier:
    """Tool to classify reaction types based on reaction SMILES and provide detailed information
    from parquet datasets instead of using LLM."""
    
    def __init__(self, dataset_path1=None, dataset_path2=None):
        """Initialize the ReactionClassifier.
        
        Args:
            dataset_path1: orderly_retro.parquet
            dataset_path2: reaction_classification_checkpoint.parquet
        """
        self.api_url = "http://13.201.135.9:9621/reaction_class"
        
        # Set default dataset paths if not provided
        self.dataset_path1 = dataset_path1
        self.dataset_path2 = dataset_path2
        
        # Initialize datasets as None
        self.dataset1 = None
        self.dataset2 = None
        
        # Try to load datasets if paths were provided
        self._try_load_datasets()
        
        # Define property mappings for common reaction properties
        self.property_mappings = {
            'temp': ['temperature', 'temp', 'reaction_temperature', 'conditions_temperature'],
            'yield': ['yield', 'reaction_yield', 'product_yield', 'yields'],
            'solvent': ['solvent', 'reaction_solvent', 'solvents'],
            'catalyst': ['catalyst', 'catalysts', 'reaction_catalyst'],
            'time': ['time', 'reaction_time', 'duration'],
            'pressure': ['pressure', 'reaction_pressure'],
            'ph': ['ph', 'reaction_ph', 'acidity']
        }
        
        # Minimum prediction certainty threshold (90%)
        self.min_certainty = 0.9
    
    def _try_load_datasets(self):
        """Try to load datasets if paths are available"""
        try:
            if self.dataset_path1 and os.path.exists(self.dataset_path1):
                self.dataset1 = pd.read_parquet(self.dataset_path1)
                print(f"Successfully loaded dataset 1: {self.dataset_path1}")
                # print(f"Dataset 1 shape: {self.dataset1.shape}")
                # print(f"Dataset 1 columns: {self.dataset1.columns.tolist()}")
            
            if self.dataset_path2 and os.path.exists(self.dataset_path2):
                self.dataset2 = pd.read_parquet(self.dataset_path2)
                print(f"Successfully loaded dataset 2: {self.dataset_path2}")
                # print(f"Dataset 2 shape: {self.dataset2.shape}")
                # print(f"Dataset 2 columns: {self.dataset2.columns.tolist()}")
                
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
    
    def _run(self, reaction_smiles, query=None):
        """Run the ReactionClassifier tool.
        
        Args:
            reaction_smiles: A reaction SMILES string
            query: Optional specific property to query (e.g., "temperature", "yield")
            
        Returns:
            A string with the classified reaction type and educational information
        """
        try:
            # Format the request payload
            payload = {"smiles": [reaction_smiles]}
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Check if we have results
                if data.get("status") == "SUCCESS" and data.get("results") and len(data["results"]) > 0:
                    # Get only the top-ranked reaction
                    top_reaction = data["results"][0]
                    reaction_name = top_reaction.get("reaction_name", "Unknown")
                    reaction_class = top_reaction.get("reaction_classname", "Unknown")
                    reaction_num = top_reaction.get("reaction_num", "Unknown")
                    certainty = top_reaction.get("prediction_certainty", 0) * 100
                    
                    # If a specific property is queried, focus on that
                    if query:
                        specific_info = self._get_specific_property(reaction_name, query)
                        if specific_info:
                            return specific_info
                    
                    result = f"## Reaction Classification\n"
                    result += f"- **Type**: {reaction_name} (Reaction #{reaction_num})\n"
                    result += f"- **Class**: {reaction_class}\n"
                    result += f"- **Certainty**: {certainty:.2f}%\n\n"
                    
                    # Get detailed information about the reaction from the datasets
                    reaction_details = self._get_reaction_info_from_datasets(reaction_name)
                    
                    # Update the message to clearly indicate the certainty threshold
                    result += f"## Detailed Information (Filtered by Prediction Certainty ≥ 90%)\n{reaction_details}\n"
                    
                    return result
                else:
                    return "No reaction classification results returned by the API."
            else:
                return f"API request failed with status code: {response.status_code}. Response: {response.text}"
        
        except Exception as e:
            return f"Error classifying reaction: {str(e)}"
    
    def _get_specific_property(self, reaction_name, property_query):
        """Get a specific property value for a reaction.
        
        Args:
            reaction_name: The name of the reaction
            property_query: The property to query (e.g., 'temperature', 'yield')
            
        Returns:
            A string with information about the specific property
        """
        try:
            if self.dataset1 is None and self.dataset2 is None:
                return f"No datasets loaded. Cannot retrieve {property_query} information."
            
            # Normalize the property query
            property_query = property_query.lower().strip()
            
            # Find which property category this query belongs to
            target_category = None
            for category, keywords in self.property_mappings.items():
                if any(keyword in property_query for keyword in keywords) or any(property_query in keyword for keyword in keywords):
                    target_category = category
                    break
            
            if not target_category:
                return f"Could not identify property type for query: '{property_query}'"
            
            # Find matches in datasets
            matches = []
            
            # Search in dataset1
            if self.dataset1 is not None:
                matches.extend(self._search_property_in_dataset(self.dataset1, reaction_name, target_category))
            
            # Search in dataset2
            if self.dataset2 is not None:
                matches.extend(self._search_property_in_dataset(self.dataset2, reaction_name, target_category))
            
            if matches:
                result = f"## {target_category.title()} Information for {reaction_name}\n\n"
                for match in matches:
                    result += f"- **{match['column']}**: {match['value']}\n"
                
                return result
            else:
                return f"No {target_category} information found for {reaction_name}."
            
        except Exception as e:
            return f"Error retrieving specific property: {str(e)}"
    
    def _search_property_in_dataset(self, dataset, reaction_name, property_category):
        """Search for a specific property in a dataset.
        
        Args:
            dataset: DataFrame containing reaction information
            reaction_name: The name of the reaction to search for
            property_category: The property category to search for
            
        Returns:
            List of matches with column names and values
        """
        matches = []
        
        try:
            # First find rows that match the reaction name
            reaction_name_cols = [col for col in dataset.columns if 'name' in col.lower() or 'reaction' in col.lower()]
            
            if not reaction_name_cols:
                return matches
            
            matching_rows = None
            
            # Search in each potential column containing reaction names
            for col in reaction_name_cols:
                # Check if the column is of string type before performing string operations
                if dataset[col].dtype == 'object':
                    # Case-insensitive matching with null-safe handling
                    col_matches = dataset[dataset[col].astype(str).str.lower().str.contains(reaction_name.lower(), na=False)]
                    
                    if not col_matches.empty:
                        matching_rows = col_matches
                        break
            
            if matching_rows is None or matching_rows.empty:
                return matches
                
            # Check for certainty column - if it exists, filter by certainty
            certainty_cols = [col for col in dataset.columns if 'certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower()]
            
            if certainty_cols:
                certainty_col = certainty_cols[0]  # Use the first certainty column found
                # Filter rows by certainty if the column exists
                if certainty_col in matching_rows.columns:
                    matching_rows = matching_rows[matching_rows[certainty_col] >= self.min_certainty]
                    
                    # If all rows were filtered out, return empty matches
                    if matching_rows.empty:
                        return matches
            
            # Now search for columns that might contain the property
            property_keywords = self.property_mappings[property_category]
            potential_columns = []
            
            for col in dataset.columns:
                col_lower = col.lower()
                # Check if column name contains any of the property keywords
                if any(keyword in col_lower for keyword in property_keywords):
                    potential_columns.append(col)
                # Also check for columns that have property in "conditions" or "details"
                elif ('condition' in col_lower or 'detail' in col_lower or 'param' in col_lower) and not pd.isna(matching_rows[col].iloc[0]):
                    # If column has JSON-like or text content, check if it contains the property
                    content = str(matching_rows[col].iloc[0]).lower()
                    if any(keyword in content for keyword in property_keywords):
                        potential_columns.append(col)
            
            # Extract the property values from each matching row
            for idx, row in matching_rows.iterrows():
                for col in potential_columns:
                    if not pd.isna(row[col]):
                        value = row[col]
                        # If the value is a string and looks like JSON or a dictionary string, 
                        # try to extract just the relevant property
                        if isinstance(value, str) and ('{' in value or ':' in value):
                            for keyword in property_keywords:
                                # Fixed regex pattern - the issue was likely here
                                try:
                                    # More robust pattern to handle various formats
                                    pattern = rf'[\'"]?{re.escape(keyword)}[\'"]?\s*[:=]\s*[\'"]?(.*?)[\'"]?(?:,|\}}|$)'
                                    match = re.search(pattern, value, re.IGNORECASE)
                                    if match:
                                        value = match.group(1).strip()
                                        break
                                except re.error as e:
                                    print(f"Regex error with keyword '{keyword}': {str(e)}")
                        
                        matches.append({
                            'column': col,
                            'value': value
                        })
        
        except Exception as e:
            print(f"Error searching property in dataset: {str(e)}")
        
        return matches
    
    def _get_reaction_info_from_datasets(self, reaction_name, prediction_certainty=0):
        """Get detailed information about a reaction type from the datasets.
        
        Args:
            reaction_name: The name of the reaction
            prediction_certainty: The certainty score from the API prediction
            
        Returns:
            A string with detailed information about the reaction
        """
        try:
            if self.dataset1 is None and self.dataset2 is None:
                return "No datasets loaded. Using only API classification information."
            
            # Use case-insensitive search for reaction name
            # Searching in dataset1
            search_result1 = self._search_reaction_in_dataset(self.dataset1, reaction_name) if self.dataset1 is not None else ""
            
            # Searching in dataset2
            search_result2 = self._search_reaction_in_dataset(self.dataset2, reaction_name) if self.dataset2 is not None else ""
            
            # Combine results from both datasets
            if search_result1 and search_result2:
                return search_result1 + "\n\n" + search_result2
            elif search_result1:
                return search_result1
            elif search_result2:
                return search_result2
            else:
                return f"No detailed information available for reaction '{reaction_name}' in the datasets with Prediction Certainty ≥ {self.min_certainty:.2%}.\n\n" + self._get_general_reaction_info(reaction_name)
                
        except Exception as e:
            return f"Error retrieving information from datasets: {str(e)}"
    
    def _search_reaction_in_dataset(self, dataset, reaction_name):
        """Search for reaction information in a specific dataset.
        
        Args:
            dataset: DataFrame containing reaction information
            reaction_name: The name of the reaction to search for
            
        Returns:
            String containing information about the reaction, or empty string if not found
        """
        try:
            if dataset is None:
                return ""
                
            # Determine which column contains reaction names based on column names
            reaction_name_cols = [col for col in dataset.columns if 'name' in col.lower() or 'reaction' in col.lower()]
            
            if not reaction_name_cols:
                return ""
            
            result = ""
            
            # Search in each potential column containing reaction names
            for col in reaction_name_cols:
                # Check if the column is of string type before performing string operations
                if dataset[col].dtype == 'object':
                    # Case-insensitive matching with null-safe handling
                    matches = dataset[dataset[col].astype(str).str.lower().str.contains(reaction_name.lower(), na=False)]
                    
                    if not matches.empty:
                        # Find prediction certainty column (specifically looking for "prediction_certainty")
                        prediction_certainty_cols = [c for c in dataset.columns if c.lower() == 'prediction_certainty' or c.lower() == 'prediction certainty']
                        
                        # If prediction certainty column doesn't exist, try more general certainty columns
                        if not prediction_certainty_cols:
                            prediction_certainty_cols = [c for c in dataset.columns if 'prediction' in c.lower() and ('certainty' in c.lower() or 'confidence' in c.lower() or 'probability' in c.lower())]
                        
                        # If still no column found, try general certainty columns
                        if not prediction_certainty_cols:
                            prediction_certainty_cols = [c for c in dataset.columns if 'certainty' in c.lower() or 'confidence' in c.lower() or 'probability' in c.lower()]
                        
                        # If prediction certainty column exists, filter by minimum certainty
                        if prediction_certainty_cols:
                            certainty_col = prediction_certainty_cols[0]  # Use the first certainty column found
                            # Filter rows by certainty threshold
                            filtered_matches = matches[matches[certainty_col] >= self.min_certainty]
                            
                            # If all rows were filtered out, return empty string
                            if filtered_matches.empty:
                                continue
                                
                            matches = filtered_matches
                        
                        # Found matching reactions with sufficient certainty, extract information
                        result += f"Found {len(matches)} entries in dataset for '{reaction_name}' with Prediction Certainty ≥ {self.min_certainty:.2%}:\n\n"
                        
                        # Process each matching row
                        for idx, row in matches.iterrows():
                            result += self._format_reaction_info(row)
                        
                        return result
            
            return ""  # No matches found in this dataset
                    
        except Exception as e:
            return f"Error searching dataset: {str(e)}"
    
    def _format_reaction_info(self, row):
        """Format the reaction information from a dataset row.
        
        Args:
            row: DataFrame row containing reaction information
            
        Returns:
            Formatted string with reaction details
        """
        result = ""
        
        # List of important fields to include (adjusted based on actual dataset columns)
        important_fields = [
            'description', 'mechanism', 'reagents', 'conditions', 
            'applications', 'limitations', 'procedure', 'temperature', 'temp',
            'yield', 'pressure', 'catalyst', 'solvent', 'time', 'ph'
        ]
        
        # Add reaction name if available
        if 'reaction_name' in row:
            result += f"### {row['reaction_name']}\n\n"
        elif 'name' in row:
            result += f"### {row['name']}\n\n"
        
        # Display prediction certainty prominently at the top of each reaction entry
        # First check for prediction_certainty column
        prediction_certainty_cols = [col for col in row.index if col.lower() == 'prediction_certainty' or col.lower() == 'prediction certainty']
        
        # If prediction certainty column doesn't exist, try more general prediction+certainty columns
        if not prediction_certainty_cols:
            prediction_certainty_cols = [col for col in row.index if 'prediction' in col.lower() and ('certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower())]
        
        # If still no column found, try general certainty columns
        if not prediction_certainty_cols:
            prediction_certainty_cols = [col for col in row.index if 'certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower()]
        
        # Display the prediction certainty value if found
        if prediction_certainty_cols:
            certainty_col = prediction_certainty_cols[0]
            certainty_value = row[certainty_col]
            if isinstance(certainty_value, (float, int)):
                if 0 <= certainty_value <= 1:  # Scale is 0-1
                    result += f"**Prediction Certainty**: {certainty_value:.4f}\n\n"
                else:  # Scale might be percentage or something else
                    result += f"**Prediction Certainty**: {certainty_value}\n\n"
            else:
                result += f"**Prediction Certainty**: {certainty_value}\n\n"
            
        # Add Original Index and Rxn Str right at the top for better organization
        if 'original_index' in row or 'index' in row:
            index_col = 'original_index' if 'original_index' in row else 'index'
            if not pd.isna(row[index_col]):
                result += f"**Original Index**: {row[index_col]}\n\n"
                
        if 'rxn_str' in row or 'reaction_smiles' in row:
            rxn_col = 'rxn_str' if 'rxn_str' in row else 'reaction_smiles'
            if not pd.isna(row[rxn_col]):
                result += f"**Rxn Str**: {row[rxn_col]}\n\n"
        
        # Add all available relevant information
        for col in row.index:
            # Check if column name contains any important field keywords
            if any(field in col.lower() for field in important_fields) and not pd.isna(row[col]):
                # Format the column name to be more readable
                formatted_col = col.replace('_', ' ').title()
                result += f"**{formatted_col}**: {row[col]}\n\n"
        
        # Add any other non-null fields that might be relevant
        # Skip prediction certainty columns as we've already displayed them at the top
        already_added_cols = important_fields + ['reaction_name', 'name', 'original_index', 'index', 'rxn_str', 'reaction_smiles']
        # We'll handle certainty columns separately now, so we don't need to skip them here
        skip_cols = ['id', 'index'] + prediction_certainty_cols
        
        for col in row.index:
            if col not in skip_cols and col not in already_added_cols and not any(field in col.lower() for field in important_fields) and not pd.isna(row[col]):
                if not col.startswith('_') and isinstance(row[col], (str, int, float)):
                    formatted_col = col.replace('_', ' ').title()
                    result += f"**{formatted_col}**: {row[col]}\n\n"
        
        return result
    
    def _get_general_reaction_info(self, reaction_name):
        """Provide general information about a reaction when no specific dataset information is available.
        
        Args:
            reaction_name: The name of the reaction
            
        Returns:
            General information about the reaction type
        """
        # Could be expanded with a dictionary of common reactions and their descriptions
        return f"The {reaction_name} is a chemical reaction identified by the classification API. " \
               f"For more detailed information about this reaction type, you may want to consult " \
               f"chemistry literature or resources."
               
    def query_reaction_property(self, reaction_smiles, property_query):
        """Query a specific property of a reaction.
        
        Args:
            reaction_smiles: A reaction SMILES string
            property_query: The property to query (e.g., "temperature", "yield")
            
        Returns:
            Information about the specified property for the given reaction
        """
        return self._run(reaction_smiles, query=property_query)






#NOW 
# import requests
# import pandas as pd
# import os
# import re

# class ReactionClassifier:
#     """Tool to classify reaction types based on reaction SMILES and provide detailed information
#     from parquet datasets instead of using LLM."""
    
#     def __init__(self, dataset_path1=None, dataset_path2=None):
#         """Initialize the ReactionClassifier.
        
#         Args:
#             dataset_path1: orderly_retro.parquet
#             dataset_path2: reaction_classification_checkpoint.parquet
#         """
#         self.api_url = "http://13.201.135.9:9621/reaction_class"
        
#         # Set default dataset paths if not provided
#         self.dataset_path1 = dataset_path1
#         self.dataset_path2 = dataset_path2
        
#         # Initialize datasets as None
#         self.dataset1 = None
#         self.dataset2 = None
        
#         # Try to load datasets if paths were provided
#         self._try_load_datasets()
        
#         # Define property mappings for common reaction properties
#         self.property_mappings = {
#             'temp': ['temperature', 'temp', 'reaction_temperature', 'conditions_temperature'],
#             'yield': ['yield', 'reaction_yield', 'product_yield', 'yields'],
#             'solvent': ['solvent', 'reaction_solvent', 'solvents'],
#             'catalyst': ['catalyst', 'catalysts', 'reaction_catalyst'],
#             'time': ['time', 'reaction_time', 'duration'],
#             'pressure': ['pressure', 'reaction_pressure'],
#             'ph': ['ph', 'reaction_ph', 'acidity'],
#             'procedure': ['procedure', 'procedure_details', 'synthesis_procedure', 'experimental_procedure']
#         }
        
#         # Minimum prediction certainty threshold (90%)
#         self.min_certainty = 0.9
    
#     def _try_load_datasets(self):
#         """Try to load datasets if paths are available"""
#         try:
#             if self.dataset_path1 and os.path.exists(self.dataset_path1):
#                 self.dataset1 = pd.read_parquet(self.dataset_path1)
#                 print(f"Successfully loaded dataset 1: {self.dataset_path1}")
#                 print(f"Dataset 1 shape: {self.dataset1.shape}")
#                 print(f"Dataset 1 columns: {self.dataset1.columns.tolist()}")
            
#             if self.dataset_path2 and os.path.exists(self.dataset_path2):
#                 self.dataset2 = pd.read_parquet(self.dataset_path2)
#                 print(f"Successfully loaded dataset 2: {self.dataset_path2}")
#                 print(f"Dataset 2 shape: {self.dataset2.shape}")
#                 print(f"Dataset 2 columns: {self.dataset2.columns.tolist()}")
                
#         except Exception as e:
#             print(f"Error loading datasets: {str(e)}")
    
#     def _run(self, reaction_smiles, query=None):
#         """Run the ReactionClassifier tool.
        
#         Args:
#             reaction_smiles: A reaction SMILES string
#             query: Optional specific property to query (e.g., "temperature", "yield")
            
#         Returns:
#             A string with the classified reaction type and educational information
#         """
#         try:
#             # Format the request payload
#             payload = {"smiles": [reaction_smiles]}
            
#             # Make the API request
#             response = requests.post(
#                 self.api_url,
#                 headers={"Content-Type": "application/json"},
#                 json=payload
#             )
            
#             # Check if request was successful
#             if response.status_code == 200:
#                 data = response.json()
                
#                 # Check if we have results
#                 if data.get("status") == "SUCCESS" and data.get("results") and len(data["results"]) > 0:
#                     # Get only the top-ranked reaction
#                     top_reaction = data["results"][0]
#                     reaction_name = top_reaction.get("reaction_name", "Unknown")
#                     reaction_class = top_reaction.get("reaction_classname", "Unknown")
#                     reaction_num = top_reaction.get("reaction_num", "Unknown")
#                     certainty = top_reaction.get("prediction_certainty", 0) * 100
                    
#                     # If a specific property is queried, focus on that
#                     if query:
#                         specific_info = self._get_specific_property(reaction_name, query)
#                         if specific_info:
#                             return specific_info
                    
#                     # Get detailed information about the reaction from the datasets
#                     reaction_details = self._get_reaction_info_from_datasets(reaction_name)
                    
#                     # Create a comprehensive output similar to test.py format
#                     result = self._create_comprehensive_report(reaction_smiles, reaction_name, reaction_class, 
#                                                             reaction_num, certainty, reaction_details)
                    
#                     return result
#                 else:
#                     return "No reaction classification results returned by the API."
#             else:
#                 return f"API request failed with status code: {response.status_code}. Response: {response.text}"
        
#         except Exception as e:
#             return f"Error classifying reaction: {str(e)}"
    
#     def _create_comprehensive_report(self, reaction_smiles, reaction_name, reaction_class, reaction_num, certainty, reaction_details):
#         """Create a comprehensive report similar to test.py format
        
#         Args:
#             reaction_smiles: The reaction SMILES string
#             reaction_name: Name of the reaction
#             reaction_class: Classification of the reaction
#             reaction_num: Reaction number
#             certainty: Prediction certainty
#             reaction_details: Detailed information from datasets
            
#         Returns:
#             A formatted comprehensive report string
#         """
#         result = f"# Comprehensive Reaction Analysis\n\n"
        
#         # Basic classification information
#         result += f"## Reaction Classification\n"
#         result += f"- **Type**: {reaction_name} (Reaction #{reaction_num})\n"
#         result += f"- **Class**: {reaction_class}\n"
#         result += f"- **Certainty**: {certainty:.2f}%\n\n"
        
#         # Reaction SMILES for reference
#         result += f"## Reaction SMILES\n"
#         result += f"`{reaction_smiles}`\n\n"
        
#         # Add detailed information from datasets
#         result += f"## Detailed Information (Filtered by Prediction Certainty ≥ 90%)\n"
#         result += reaction_details
        
#         # Summary section at the end
#         result += f"\n## Summary\n"
#         result += f"This is a {reaction_name} reaction, classified as {reaction_class} with {certainty:.2f}% certainty. "
        
#         # Add a note about bond formation based on the reaction class
#         if "C-C bond formation" in reaction_class:
#             result += "This reaction involves the formation of carbon-carbon bonds, which is fundamental in organic synthesis for building molecular frameworks. "
#         elif "Organometallic" in reaction_class:
#             result += "This reaction utilizes organometallic reagents, which often provide unique reactivity patterns not achievable with purely organic reagents. "
        
#         result += "For more detailed information about mechanism and applications, please refer to the detailed information section above."
        
#         return result
    
#     def _get_specific_property(self, reaction_name, property_query):
#         """Get a specific property value for a reaction.
        
#         Args:
#             reaction_name: The name of the reaction
#             property_query: The property to query (e.g., 'temperature', 'yield')
            
#         Returns:
#             A string with information about the specific property
#         """
#         try:
#             if self.dataset1 is None and self.dataset2 is None:
#                 return f"No datasets loaded. Cannot retrieve {property_query} information."
            
#             # Normalize the property query
#             property_query = property_query.lower().strip()
            
#             # Find which property category this query belongs to
#             target_category = None
#             for category, keywords in self.property_mappings.items():
#                 if any(keyword in property_query for keyword in keywords) or any(property_query in keyword for keyword in keywords):
#                     target_category = category
#                     break
            
#             if not target_category:
#                 return f"Could not identify property type for query: '{property_query}'"
            
#             # Find matches in datasets
#             matches = []
            
#             # Search in dataset1
#             if self.dataset1 is not None:
#                 matches.extend(self._search_property_in_dataset(self.dataset1, reaction_name, target_category))
            
#             # Search in dataset2
#             if self.dataset2 is not None:
#                 matches.extend(self._search_property_in_dataset(self.dataset2, reaction_name, target_category))
            
#             if matches:
#                 result = f"## {target_category.title()} Information for {reaction_name}\n\n"
#                 for match in matches:
#                     result += f"- **{match['column']}**: {match['value']}\n"
                
#                 return result
#             else:
#                 return f"No {target_category} information found for {reaction_name}."
            
#         except Exception as e:
#             return f"Error retrieving specific property: {str(e)}"
    
#     def _search_property_in_dataset(self, dataset, reaction_name, property_category):
#         """Search for a specific property in a dataset.
        
#         Args:
#             dataset: DataFrame containing reaction information
#             reaction_name: The name of the reaction to search for
#             property_category: The property category to search for
            
#         Returns:
#             List of matches with column names and values
#         """
#         matches = []
        
#         try:
#             # First find rows that match the reaction name
#             reaction_name_cols = [col for col in dataset.columns if 'name' in col.lower() or 'reaction' in col.lower()]
            
#             if not reaction_name_cols:
#                 return matches
            
#             matching_rows = None
            
#             # Search in each potential column containing reaction names
#             for col in reaction_name_cols:
#                 # Check if the column is of string type before performing string operations
#                 if dataset[col].dtype == 'object':
#                     # Case-insensitive matching with null-safe handling
#                     col_matches = dataset[dataset[col].astype(str).str.lower().str.contains(reaction_name.lower(), na=False)]
                    
#                     if not col_matches.empty:
#                         matching_rows = col_matches
#                         break
            
#             if matching_rows is None or matching_rows.empty:
#                 return matches
                
#             # Check for certainty column - if it exists, filter by certainty
#             certainty_cols = [col for col in dataset.columns if 'certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower()]
            
#             if certainty_cols:
#                 certainty_col = certainty_cols[0]  # Use the first certainty column found
#                 # Filter rows by certainty if the column exists
#                 if certainty_col in matching_rows.columns:
#                     matching_rows = matching_rows[matching_rows[certainty_col] >= self.min_certainty]
                    
#                     # If all rows were filtered out, return empty matches
#                     if matching_rows.empty:
#                         return matches
            
#             # Now search for columns that might contain the property
#             property_keywords = self.property_mappings[property_category]
#             potential_columns = []
            
#             for col in dataset.columns:
#                 col_lower = col.lower()
#                 # Check if column name contains any of the property keywords
#                 if any(keyword in col_lower for keyword in property_keywords):
#                     potential_columns.append(col)
#                 # Also check for columns that have property in "conditions" or "details"
#                 elif ('condition' in col_lower or 'detail' in col_lower or 'param' in col_lower) and not pd.isna(matching_rows[col].iloc[0]):
#                     # If column has JSON-like or text content, check if it contains the property
#                     content = str(matching_rows[col].iloc[0]).lower()
#                     if any(keyword in content for keyword in property_keywords):
#                         potential_columns.append(col)
            
#             # Extract the property values from each matching row
#             for idx, row in matching_rows.iterrows():
#                 for col in potential_columns:
#                     if not pd.isna(row[col]):
#                         value = row[col]
#                         # If the value is a string and looks like JSON or a dictionary string, 
#                         # try to extract just the relevant property
#                         if isinstance(value, str) and ('{' in value or ':' in value):
#                             for keyword in property_keywords:
#                                 # Fixed regex pattern - the issue was likely here
#                                 try:
#                                     # More robust pattern to handle various formats
#                                     pattern = rf'[\'"]?{re.escape(keyword)}[\'"]?\s*[:=]\s*[\'"]?(.*?)[\'"]?(?:,|\}}|$)'
#                                     match = re.search(pattern, value, re.IGNORECASE)
#                                     if match:
#                                         value = match.group(1).strip()
#                                         break
#                                 except re.error as e:
#                                     print(f"Regex error with keyword '{keyword}': {str(e)}")
                        
#                         matches.append({
#                             'column': col,
#                             'value': value
#                         })
        
#         except Exception as e:
#             print(f"Error searching property in dataset: {str(e)}")
        
#         return matches
    
#     def _get_reaction_info_from_datasets(self, reaction_name, prediction_certainty=0):
#         """Get detailed information about a reaction type from the datasets.
        
#         Args:
#             reaction_name: The name of the reaction
#             prediction_certainty: The certainty score from the API prediction
            
#         Returns:
#             A string with detailed information about the reaction
#         """
#         try:
#             if self.dataset1 is None and self.dataset2 is None:
#                 return "No datasets loaded. Using only API classification information."
            
#             # Use case-insensitive search for reaction name
#             # Searching in dataset1
#             search_result1 = self._search_reaction_in_dataset(self.dataset1, reaction_name) if self.dataset1 is not None else ""
            
#             # Searching in dataset2
#             search_result2 = self._search_reaction_in_dataset(self.dataset2, reaction_name) if self.dataset2 is not None else ""
            
#             # Combine results from both datasets
#             if search_result1 and search_result2:
#                 return search_result1 + "\n\n" + search_result2
#             elif search_result1:
#                 return search_result1
#             elif search_result2:
#                 return search_result2
#             else:
#                 return f"No detailed information available for reaction '{reaction_name}' in the datasets with Prediction Certainty ≥ {self.min_certainty:.2%}.\n\n" + self._get_general_reaction_info(reaction_name)
                
#         except Exception as e:
#             return f"Error retrieving information from datasets: {str(e)}"
    
#     def _search_reaction_in_dataset(self, dataset, reaction_name):
#         """Search for reaction information in a specific dataset.
        
#         Args:
#             dataset: DataFrame containing reaction information
#             reaction_name: The name of the reaction to search for
            
#         Returns:
#             String containing information about the reaction, or empty string if not found
#         """
#         try:
#             if dataset is None:
#                 return ""
                
#             # Determine which column contains reaction names based on column names
#             reaction_name_cols = [col for col in dataset.columns if 'name' in col.lower() or 'reaction' in col.lower()]
            
#             if not reaction_name_cols:
#                 return ""
            
#             result = ""
            
#             # Search in each potential column containing reaction names
#             for col in reaction_name_cols:
#                 # Check if the column is of string type before performing string operations
#                 if dataset[col].dtype == 'object':
#                     # Case-insensitive matching with null-safe handling
#                     matches = dataset[dataset[col].astype(str).str.lower().str.contains(reaction_name.lower(), na=False)]
                    
#                     if not matches.empty:
#                         # Find prediction certainty column (specifically looking for "prediction_certainty")
#                         prediction_certainty_cols = [c for c in dataset.columns if c.lower() == 'prediction_certainty' or c.lower() == 'prediction certainty']
                        
#                         # If prediction certainty column doesn't exist, try more general certainty columns
#                         if not prediction_certainty_cols:
#                             prediction_certainty_cols = [c for c in dataset.columns if 'prediction' in c.lower() and ('certainty' in c.lower() or 'confidence' in c.lower() or 'probability' in c.lower())]
                        
#                         # If still no column found, try general certainty columns
#                         if not prediction_certainty_cols:
#                             prediction_certainty_cols = [c for c in dataset.columns if 'certainty' in c.lower() or 'confidence' in c.lower() or 'probability' in c.lower()]
                        
#                         # If prediction certainty column exists, filter by minimum certainty
#                         if prediction_certainty_cols:
#                             certainty_col = prediction_certainty_cols[0]  # Use the first certainty column found
#                             # Filter rows by certainty threshold
#                             filtered_matches = matches[matches[certainty_col] >= self.min_certainty]
                            
#                             # If all rows were filtered out, return empty string
#                             if filtered_matches.empty:
#                                 continue
                                
#                             matches = filtered_matches
                        
#                         # Found matching reactions with sufficient certainty, extract information
#                         result += f"Found {len(matches)} entries in dataset for '{reaction_name}' with Prediction Certainty ≥ {self.min_certainty:.2%}:\n\n"
                        
#                         # Process each matching row
#                         for idx, row in matches.iterrows():
#                             result += self._format_reaction_info(row)
                        
#                         return result
            
#             return ""  # No matches found in this dataset
                    
#         except Exception as e:
#             return f"Error searching dataset: {str(e)}"
    
#     def _format_reaction_info(self, row):
#         """Format the reaction information from a dataset row.
        
#         Args:
#             row: DataFrame row containing reaction information
            
#         Returns:
#             Formatted string with reaction details
#         """
#         result = ""
        
#         # List of important fields to include (adjusted based on actual dataset columns)
#         important_fields = [
#             'description', 'mechanism', 'reagents', 'conditions', 
#             'applications', 'limitations', 'procedure', 'procedure_details',
#             'temperature', 'temp', 'yield', 'pressure', 'catalyst', 'solvent', 
#             'time', 'ph', 'rxn_time'
#         ]
        
#         # Add reaction name if available
#         if 'reaction_name' in row:
#             result += f"### {row['reaction_name']}\n\n"
#         elif 'name' in row:
#             result += f"### {row['name']}\n\n"
        
#         # Display prediction certainty prominently at the top of each reaction entry
#         # First check for prediction_certainty column
#         prediction_certainty_cols = [col for col in row.index if col.lower() == 'prediction_certainty' or col.lower() == 'prediction certainty']
        
#         # If prediction certainty column doesn't exist, try more general prediction+certainty columns
#         if not prediction_certainty_cols:
#             prediction_certainty_cols = [col for col in row.index if 'prediction' in col.lower() and ('certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower())]
        
#         # If still no column found, try general certainty columns
#         if not prediction_certainty_cols:
#             prediction_certainty_cols = [col for col in row.index if 'certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower()]
        
#         # Display the prediction certainty value if found
#         if prediction_certainty_cols:
#             certainty_col = prediction_certainty_cols[0]
#             certainty_value = row[certainty_col]
#             if isinstance(certainty_value, (float, int)):
#                 if 0 <= certainty_value <= 1:  # Scale is 0-1
#                     result += f"**Prediction Certainty**: {certainty_value:.4f}\n\n"
#                 else:  # Scale might be percentage or something else
#                     result += f"**Prediction Certainty**: {certainty_value}\n\n"
#             else:
#                 result += f"**Prediction Certainty**: {certainty_value}\n\n"
            
#         # Add Original Index and Rxn Str right at the top for better organization
#         if 'original_index' in row or 'index' in row:
#             index_col = 'original_index' if 'original_index' in row else 'index'
#             if not pd.isna(row[index_col]):
#                 result += f"**Original Index**: {row[index_col]}\n\n"
                
#         if 'rxn_str' in row or 'reaction_smiles' in row:
#             rxn_col = 'rxn_str' if 'rxn_str' in row else 'reaction_smiles'
#             if not pd.isna(row[rxn_col]):
#                 result += f"**Rxn Str**: {row[rxn_col]}\n\n"
        
#         # IMPORTANT: Add procedure_details first with special formatting if it exists
#         if 'procedure_details' in row and not pd.isna(row['procedure_details']):
#             result += f"### Procedure Details\n"
#             procedure = str(row['procedure_details'])
#             # Format procedure steps if they appear to be numbered
#             if re.search(r'^\d+\.', procedure):
#                 steps = re.split(r'(\d+\.)', procedure)
#                 formatted_procedure = ""
#                 for i in range(1, len(steps), 2):
#                     if i+1 < len(steps):
#                         formatted_procedure += f"{steps[i]}{steps[i+1]}\n"
#                 result += formatted_procedure + "\n"
#             else:
#                 result += f"{procedure}\n\n"
        
#         # Add all available relevant information
#         for col in row.index:
#             # Check if column name contains any important field keywords
#             if any(field in col.lower() for field in important_fields) and not pd.isna(row[col]) and col != 'procedure_details':  # Skip procedure_details as we've handled it separately
#                 # Format the column name to be more readable
#                 formatted_col = col.replace('_', ' ').title()
#                 result += f"**{formatted_col}**: {row[col]}\n\n"
        
#         # Add any other non-null fields that might be relevant
#         # Skip prediction certainty columns as we've already displayed them at the top
#         already_added_cols = important_fields + ['reaction_name', 'name', 'original_index', 'index', 'rxn_str', 'reaction_smiles']
#         # We'll handle certainty columns separately now, so we don't need to skip them here
#         skip_cols = ['id', 'index'] + prediction_certainty_cols
        
#         for col in row.index:
#             if col not in skip_cols and col not in already_added_cols and not any(field in col.lower() for field in important_fields) and not pd.isna(row[col]):
#                 if not col.startswith('_') and isinstance(row[col], (str, int, float)):
#                     formatted_col = col.replace('_', ' ').title()
#                     result += f"**{formatted_col}**: {row[col]}\n\n"
        
#         return result
    
#     def _get_general_reaction_info(self, reaction_name):
#         """Provide general information about a reaction when no specific dataset information is available.
        
#         Args:
#             reaction_name: The name of the reaction
            
#         Returns:
#             General information about the reaction type
#         """
#         # Dictionary of common reactions with educational descriptions
#         common_reactions = {
#             "Weinreb iodo coupling": "The Weinreb iodo coupling is an organometallic C-C bond formation reaction. It typically involves the reaction of an iodide with a Weinreb amide in the presence of a metal catalyst to form a new carbon-carbon bond. This reaction is valuable in organic synthesis for constructing complex molecular frameworks.",
#             "Suzuki coupling": "The Suzuki coupling is a palladium-catalyzed cross-coupling reaction between organoboron compounds and organohalides. It is widely used for C-C bond formation in pharmaceutical and materials science applications.",
#             "Heck reaction": "The Heck reaction is a palladium-catalyzed coupling reaction between an unsaturated halide and an alkene in the presence of a base to form a substituted alkene. This reaction is important for constructing complex carbon frameworks.",
#             "Buchwald-Hartwig amination": "The Buchwald-Hartwig amination is a palladium-catalyzed coupling reaction that forms C-N bonds between aryl halides or triflates and amines. This reaction is widely used in the synthesis of pharmaceuticals containing nitrogen."
#         }
        
#         if reaction_name in common_reactions:
#             return common_reactions[reaction_name]
#         else:
#             return f"The {reaction_name} is a chemical reaction identified by the classification API. " \
#                    f"For more detailed information about this reaction type, you may want to consult " \
#                    f"chemistry literature or resources."
               
#     def query_reaction_property(self, reaction_smiles, property_query):
#         """Query a specific property of a reaction.
        
#         Args:
#             reaction_smiles: A reaction SMILES string
#             property_query: The property to query (e.g., "temperature", "yield")
            
#         Returns:
#             Information about the specified property for the given reaction
#         """
#         return self._run(reaction_smiles, query=property_query)




# import requests
# import pandas as pd
# import os
# import re

# class ReactionClassifier:
#     """Tool to classify reaction types based on reaction SMILES and provide detailed information
#     from parquet datasets instead of using LLM."""
    
#     def __init__(self, dataset_path1=None, dataset_path2=None):
#         """Initialize the ReactionClassifier.
        
#         Args:
#             dataset_path1: orderly_retro.parquet
#             dataset_path2: reaction_classification_checkpoint.parquet
#         """
#         self.api_url = "http://13.201.135.9:9621/reaction_class"
        
#         # Set default dataset paths if not provided
#         self.dataset_path1 = dataset_path1
#         self.dataset_path2 = dataset_path2
        
#         # Initialize datasets as None
#         self.dataset1 = None
#         self.dataset2 = None
        
#         # Try to load datasets if paths were provided
#         self._try_load_datasets()
        
#         # Define property mappings for common reaction properties
#         self.property_mappings = {
#             'temp': ['temperature', 'temp', 'reaction_temperature', 'conditions_temperature'],
#             'yield': ['yield', 'reaction_yield', 'product_yield', 'yields'],
#             'solvent': ['solvent', 'reaction_solvent', 'solvents'],
#             'catalyst': ['catalyst', 'catalysts', 'reaction_catalyst'],
#             'time': ['time', 'reaction_time', 'duration'],
#             'pressure': ['pressure', 'reaction_pressure'],
#             'ph': ['ph', 'reaction_ph', 'acidity']
#         }
        
#         # Minimum prediction certainty threshold (90%)
#         self.min_certainty = 0.9
    
#     def _try_load_datasets(self):
#         """Try to load datasets if paths are available"""
#         try:
#             if self.dataset_path1 and os.path.exists(self.dataset_path1):
#                 self.dataset1 = pd.read_parquet(self.dataset_path1)
#                 print(f"Successfully loaded dataset 1: {self.dataset_path1}")
#                 print(f"Dataset 1 shape: {self.dataset1.shape}")
#                 print(f"Dataset 1 columns: {self.dataset1.columns.tolist()}")
            
#             if self.dataset_path2 and os.path.exists(self.dataset_path2):
#                 self.dataset2 = pd.read_parquet(self.dataset_path2)
#                 print(f"Successfully loaded dataset 2: {self.dataset_path2}")
#                 print(f"Dataset 2 shape: {self.dataset2.shape}")
#                 print(f"Dataset 2 columns: {self.dataset2.columns.tolist()}")
                
#         except Exception as e:
#             print(f"Error loading datasets: {str(e)}")
    
#     def _run(self, reaction_smiles, query=None):
#         """Run the ReactionClassifier tool.
        
#         Args:
#             reaction_smiles: A reaction SMILES string
#             query: Optional specific property to query (e.g., "temperature", "yield")
            
#         Returns:
#             A string with the classified reaction type and educational information
#         """
#         try:
#             # Format the request payload
#             payload = {"smiles": [reaction_smiles]}
            
#             # Make the API request
#             response = requests.post(
#                 self.api_url,
#                 headers={"Content-Type": "application/json"},
#                 json=payload
#             )
            
#             # Check if request was successful
#             if response.status_code == 200:
#                 data = response.json()
                
#                 # Check if we have results
#                 if data.get("status") == "SUCCESS" and data.get("results") and len(data["results"]) > 0:
#                     # Get only the top-ranked reaction
#                     top_reaction = data["results"][0]
#                     reaction_name = top_reaction.get("reaction_name", "Unknown")
#                     reaction_class = top_reaction.get("reaction_classname", "Unknown")
#                     reaction_num = top_reaction.get("reaction_num", "Unknown")
#                     certainty = top_reaction.get("prediction_certainty", 0) * 100
                    
#                     # If a specific property is queried, focus on that
#                     if query:
#                         specific_info = self._get_specific_property(reaction_name, query)
#                         if specific_info:
#                             return specific_info
                    
#                     result = f"## Reaction Classification\n"
#                     result += f"- **Type**: {reaction_name} (Reaction #{reaction_num})\n"
#                     result += f"- **Class**: {reaction_class}\n"
#                     result += f"- **Certainty**: {certainty:.2f}%\n\n"
                    
#                     # Get detailed information about the reaction from the datasets
#                     reaction_details = self._get_reaction_info_from_datasets(reaction_name)
                    
#                     # Update the message to clearly indicate the certainty threshold
#                     result += f"## Detailed Information (Filtered by Prediction Certainty ≥ 90%)\n{reaction_details}\n"
                    
#                     return result
#                 else:
#                     return "No reaction classification results returned by the API."
#             else:
#                 return f"API request failed with status code: {response.status_code}. Response: {response.text}"
        
#         except Exception as e:
#             return f"Error classifying reaction: {str(e)}"
    
#     def _get_specific_property(self, reaction_name, property_query):
#         """Get a specific property value for a reaction.
        
#         Args:
#             reaction_name: The name of the reaction
#             property_query: The property to query (e.g., 'temperature', 'yield')
            
#         Returns:
#             A string with information about the specific property
#         """
#         try:
#             if self.dataset1 is None and self.dataset2 is None:
#                 return f"No datasets loaded. Cannot retrieve {property_query} information."
            
#             # Normalize the property query
#             property_query = property_query.lower().strip()
            
#             # Find which property category this query belongs to
#             target_category = None
#             for category, keywords in self.property_mappings.items():
#                 if any(keyword in property_query for keyword in keywords) or any(property_query in keyword for keyword in keywords):
#                     target_category = category
#                     break
            
#             if not target_category:
#                 return f"Could not identify property type for query: '{property_query}'"
            
#             # Find matches in datasets
#             matches = []
            
#             # Search in dataset1
#             if self.dataset1 is not None:
#                 matches.extend(self._search_property_in_dataset(self.dataset1, reaction_name, target_category))
            
#             # Search in dataset2
#             if self.dataset2 is not None:
#                 matches.extend(self._search_property_in_dataset(self.dataset2, reaction_name, target_category))
            
#             if matches:
#                 result = f"## {target_category.title()} Information for {reaction_name}\n\n"
#                 for match in matches:
#                     result += f"- **{match['column']}**: {match['value']}\n"
                
#                 return result
#             else:
#                 return f"No {target_category} information found for {reaction_name}."
            
#         except Exception as e:
#             return f"Error retrieving specific property: {str(e)}"
    
#     def _search_property_in_dataset(self, dataset, reaction_name, property_category):
#         """Search for a specific property in a dataset.
        
#         Args:
#             dataset: DataFrame containing reaction information
#             reaction_name: The name of the reaction to search for
#             property_category: The property category to search for
            
#         Returns:
#             List of matches with column names and values
#         """
#         matches = []
        
#         try:
#             # First find rows that match the reaction name
#             reaction_name_cols = [col for col in dataset.columns if 'name' in col.lower() or 'reaction' in col.lower()]
            
#             if not reaction_name_cols:
#                 return matches
            
#             matching_rows = None
            
#             # Search in each potential column containing reaction names
#             for col in reaction_name_cols:
#                 # Check if the column is of string type before performing string operations
#                 if dataset[col].dtype == 'object':
#                     # Case-insensitive matching with null-safe handling
#                     col_matches = dataset[dataset[col].astype(str).str.lower().str.contains(reaction_name.lower(), na=False)]
                    
#                     if not col_matches.empty:
#                         matching_rows = col_matches
#                         break
            
#             if matching_rows is None or matching_rows.empty:
#                 return matches
                
#             # Check for certainty column - if it exists, filter by certainty
#             certainty_cols = [col for col in dataset.columns if 'certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower()]
            
#             if certainty_cols:
#                 certainty_col = certainty_cols[0]  # Use the first certainty column found
#                 # Filter rows by certainty if the column exists
#                 if certainty_col in matching_rows.columns:
#                     matching_rows = matching_rows[matching_rows[certainty_col] >= self.min_certainty]
                    
#                     # If all rows were filtered out, return empty matches
#                     if matching_rows.empty:
#                         return matches
            
#             # Now search for columns that might contain the property
#             property_keywords = self.property_mappings[property_category]
#             potential_columns = []
            
#             for col in dataset.columns:
#                 col_lower = col.lower()
#                 # Check if column name contains any of the property keywords
#                 if any(keyword in col_lower for keyword in property_keywords):
#                     potential_columns.append(col)
#                 # Also check for columns that have property in "conditions" or "details"
#                 elif ('condition' in col_lower or 'detail' in col_lower or 'param' in col_lower) and not pd.isna(matching_rows[col].iloc[0]):
#                     # If column has JSON-like or text content, check if it contains the property
#                     content = str(matching_rows[col].iloc[0]).lower()
#                     if any(keyword in content for keyword in property_keywords):
#                         potential_columns.append(col)
            
#             # Extract the property values from each matching row
#             for idx, row in matching_rows.iterrows():
#                 for col in potential_columns:
#                     if not pd.isna(row[col]):
#                         value = row[col]
#                         # If the value is a string and looks like JSON or a dictionary string, 
#                         # try to extract just the relevant property
#                         if isinstance(value, str) and ('{' in value or ':' in value):
#                             for keyword in property_keywords:
#                                 # Fixed regex pattern - the issue was likely here
#                                 try:
#                                     # More robust pattern to handle various formats
#                                     pattern = rf'[\'"]?{re.escape(keyword)}[\'"]?\s*[:=]\s*[\'"]?(.*?)[\'"]?(?:,|\}}|$)'
#                                     match = re.search(pattern, value, re.IGNORECASE)
#                                     if match:
#                                         value = match.group(1).strip()
#                                         break
#                                 except re.error as e:
#                                     print(f"Regex error with keyword '{keyword}': {str(e)}")
                        
#                         matches.append({
#                             'column': col,
#                             'value': value
#                         })
        
#         except Exception as e:
#             print(f"Error searching property in dataset: {str(e)}")
        
#         return matches
    
#     def _get_reaction_info_from_datasets(self, reaction_name, prediction_certainty=0):
#         """Get detailed information about a reaction type from the datasets.
        
#         Args:
#             reaction_name: The name of the reaction
#             prediction_certainty: The certainty score from the API prediction
            
#         Returns:
#             A string with detailed information about the reaction
#         """
#         try:
#             if self.dataset1 is None and self.dataset2 is None:
#                 return "No datasets loaded. Using only API classification information."
            
#             # Use case-insensitive search for reaction name
#             # Searching in dataset1
#             search_result1 = self._search_reaction_in_dataset(self.dataset1, reaction_name) if self.dataset1 is not None else ""
            
#             # Searching in dataset2
#             search_result2 = self._search_reaction_in_dataset(self.dataset2, reaction_name) if self.dataset2 is not None else ""
            
#             # Combine results from both datasets
#             if search_result1 and search_result2:
#                 return search_result1 + "\n\n" + search_result2
#             elif search_result1:
#                 return search_result1
#             elif search_result2:
#                 return search_result2
#             else:
#                 return f"No detailed information available for reaction '{reaction_name}' in the datasets with Prediction Certainty ≥ {self.min_certainty:.2%}.\n\n" + self._get_general_reaction_info(reaction_name)
                
#         except Exception as e:
#             return f"Error retrieving information from datasets: {str(e)}"
    
#     def _search_reaction_in_dataset(self, dataset, reaction_name):
#         """Search for reaction information in a specific dataset.
        
#         Args:
#             dataset: DataFrame containing reaction information
#             reaction_name: The name of the reaction to search for
            
#         Returns:
#             String containing information about the reaction, or empty string if not found
#         """
#         try:
#             if dataset is None:
#                 return ""
                
#             # Determine which column contains reaction names based on column names
#             reaction_name_cols = [col for col in dataset.columns if 'name' in col.lower() or 'reaction' in col.lower()]
            
#             if not reaction_name_cols:
#                 return ""
            
#             result = ""
            
#             # Search in each potential column containing reaction names
#             for col in reaction_name_cols:
#                 # Check if the column is of string type before performing string operations
#                 if dataset[col].dtype == 'object':
#                     # Case-insensitive matching with null-safe handling
#                     matches = dataset[dataset[col].astype(str).str.lower().str.contains(reaction_name.lower(), na=False)]
                    
#                     if not matches.empty:
#                         # Find prediction certainty column (specifically looking for "prediction_certainty")
#                         prediction_certainty_cols = [c for c in dataset.columns if c.lower() == 'prediction_certainty' or c.lower() == 'prediction certainty']
                        
#                         # If prediction certainty column doesn't exist, try more general certainty columns
#                         if not prediction_certainty_cols:
#                             prediction_certainty_cols = [c for c in dataset.columns if 'prediction' in c.lower() and ('certainty' in c.lower() or 'confidence' in c.lower() or 'probability' in c.lower())]
                        
#                         # If still no column found, try general certainty columns
#                         if not prediction_certainty_cols:
#                             prediction_certainty_cols = [c for c in dataset.columns if 'certainty' in c.lower() or 'confidence' in c.lower() or 'probability' in c.lower()]
                        
#                         # If prediction certainty column exists, filter by minimum certainty
#                         if prediction_certainty_cols:
#                             certainty_col = prediction_certainty_cols[0]  # Use the first certainty column found
#                             # Filter rows by certainty threshold
#                             filtered_matches = matches[matches[certainty_col] >= self.min_certainty]
                            
#                             # If all rows were filtered out, return empty string
#                             if filtered_matches.empty:
#                                 continue
                                
#                             matches = filtered_matches
                        
#                         # Found matching reactions with sufficient certainty, extract information
#                         result += f"Found {len(matches)} entries in dataset for '{reaction_name}' with Prediction Certainty ≥ {self.min_certainty:.2%}:\n\n"
                        
#                         # Process each matching row
#                         for idx, row in matches.iterrows():
#                             result += self._format_reaction_info(row)
                        
#                         return result
            
#             return ""  # No matches found in this dataset
                    
#         except Exception as e:
#             return f"Error searching dataset: {str(e)}"
    
#     def _format_reaction_info(self, row):
#         """Format the reaction information from a dataset row.
        
#         Args:
#             row: DataFrame row containing reaction information
            
#         Returns:
#             Formatted string with reaction details
#         """
#         result = ""
        
#         # List of important fields to include (adjusted based on actual dataset columns)
#         important_fields = [
#             'description', 'mechanism', 'reagents', 'conditions', 
#             'applications', 'limitations', 'procedure', 'temperature', 'temp',
#             'yield', 'pressure', 'catalyst', 'solvent', 'time', 'ph'
#         ]
        
#         # Add reaction name if available
#         if 'reaction_name' in row:
#             result += f"### {row['reaction_name']}\n\n"
#         elif 'name' in row:
#             result += f"### {row['name']}\n\n"
        
#         # Display prediction certainty prominently at the top of each reaction entry
#         # First check for prediction_certainty column
#         prediction_certainty_cols = [col for col in row.index if col.lower() == 'prediction_certainty' or col.lower() == 'prediction certainty']
        
#         # If prediction certainty column doesn't exist, try more general prediction+certainty columns
#         if not prediction_certainty_cols:
#             prediction_certainty_cols = [col for col in row.index if 'prediction' in col.lower() and ('certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower())]
        
#         # If still no column found, try general certainty columns
#         if not prediction_certainty_cols:
#             prediction_certainty_cols = [col for col in row.index if 'certainty' in col.lower() or 'confidence' in col.lower() or 'probability' in col.lower()]
        
#         # Display the prediction certainty value if found
#         if prediction_certainty_cols:
#             certainty_col = prediction_certainty_cols[0]
#             certainty_value = row[certainty_col]
#             if isinstance(certainty_value, (float, int)):
#                 if 0 <= certainty_value <= 1:  # Scale is 0-1
#                     result += f"**Prediction Certainty**: {certainty_value:.4f}\n\n"
#                 else:  # Scale might be percentage or something else
#                     result += f"**Prediction Certainty**: {certainty_value}\n\n"
#             else:
#                 result += f"**Prediction Certainty**: {certainty_value}\n\n"
            
#         # Add Original Index and Rxn Str right at the top for better organization
#         if 'original_index' in row or 'index' in row:
#             index_col = 'original_index' if 'original_index' in row else 'index'
#             if not pd.isna(row[index_col]):
#                 result += f"**Original Index**: {row[index_col]}\n\n"
                
#         if 'rxn_str' in row or 'reaction_smiles' in row:
#             rxn_col = 'rxn_str' if 'rxn_str' in row else 'reaction_smiles'
#             if not pd.isna(row[rxn_col]):
#                 result += f"**Rxn Str**: {row[rxn_col]}\n\n"
        
#         # Add all available relevant information
#         for col in row.index:
#             # Check if column name contains any important field keywords
#             if any(field in col.lower() for field in important_fields) and not pd.isna(row[col]):
#                 # Format the column name to be more readable
#                 formatted_col = col.replace('_', ' ').title()
#                 result += f"**{formatted_col}**: {row[col]}\n\n"
        
#         # Add any other non-null fields that might be relevant
#         # Skip prediction certainty columns as we've already displayed them at the top
#         already_added_cols = important_fields + ['reaction_name', 'name', 'original_index', 'index', 'rxn_str', 'reaction_smiles']
#         # We'll handle certainty columns separately now, so we don't need to skip them here
#         skip_cols = ['id', 'index'] + prediction_certainty_cols
        
#         for col in row.index:
#             if col not in skip_cols and col not in already_added_cols and not any(field in col.lower() for field in important_fields) and not pd.isna(row[col]):
#                 if not col.startswith('_') and isinstance(row[col], (str, int, float)):
#                     formatted_col = col.replace('_', ' ').title()
#                     result += f"**{formatted_col}**: {row[col]}\n\n"
        
#         return result
    
#     def _get_general_reaction_info(self, reaction_name):
#         """Provide general information about a reaction when no specific dataset information is available.
        
#         Args:
#             reaction_name: The name of the reaction
            
#         Returns:
#             General information about the reaction type
#         """
#         # Could be expanded with a dictionary of common reactions and their descriptions
#         return f"The {reaction_name} is a chemical reaction identified by the classification API. " \
#                f"For more detailed information about this reaction type, you may want to consult " \
#                f"chemistry literature or resources."
               
#     def query_reaction_property(self, reaction_smiles, property_query):
#         """Query a specific property of a reaction.
        
#         Args:
#             reaction_smiles: A reaction SMILES string
#             property_query: The property to query (e.g., "temperature", "yield")
            
#         Returns:
#             Information about the specified property for the given reaction
#         """
#         return self._run(reaction_smiles, query=property_query)