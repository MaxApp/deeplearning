import re
import matplotlib.pyplot as plt
import random
import pandas as pd
import ast

def clean_and_tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    # data = nltk.word_tokenize(data)
    data = [ch.lower() for ch in data]
    return data

def preprocess_text(text):
    text = text.lower()
    # remove all characters that are not letters or whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # split into a list of words
    words = text.split()
    return words

def get_sliding_context(tokenized_words: list[str], half_context_size: int):
    """A sliding window generator"""
    i = half_context_size
    while i < len(tokenized_words) - half_context_size:
        center_word = tokenized_words[i]
        context_words = tokenized_words[(i - half_context_size):i] + tokenized_words[(i+1):(i+half_context_size+1)]
        yield context_words, center_word
        i += 1

def plot_embeddings(coords, labels, label_dict, title):
    """
    Visualizes 2D word embeddings using a scatter plot.

    This function plots a set of 2D coordinates, annotates each point with
    a corresponding label, and colors the points based on predefined
    categories. It dynamically assigns colors to categories for clear
    visual distinction.

    Args:
        coords: A 2D numpy array where each row represents the x, y
                coordinates of a point.
        labels: A list of string labels, one for each point in `coords`.
        label_dict: A dictionary that maps category names to lists of
                    words belonging to that category.
        title: The title for the plot.
    """
    # Use a try-except block to handle potential inconsistencies
    # between the labels and the dictionary.
    try:
        # Create a reverse mapping from words to categories for efficient lookup.
        word_to_category = {word: category for category, words in label_dict.items() for word in words}

        # Validate that every label has a corresponding category in the map.
        # This will raise a KeyError if a word is not found.
        for word in labels:
            _ = word_to_category[word]

        # --- Dynamic Color Assignment ---
        # Define a base list of visually distinct colors for consistency.
        fixed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Use a set for efficient tracking of assigned colors to prevent duplicates.
        used_colors = set(fixed_colors)
        
        # Get a list of all unique category names.
        unique_categories = list(label_dict.keys())
        # Initialize a dictionary to map each category to a unique color.
        category_to_color = {}

        # Iterate through each unique category to assign a color.
        for i, category in enumerate(unique_categories):
            # Assign colors from the predefined list first.
            if i < len(fixed_colors):
                category_to_color[category] = fixed_colors[i]
            else:
                # Generate a unique random color for any additional categories.
                random_color = None
                while random_color is None or random_color in used_colors:
                    # Generate a random hex color string (e.g., '#a1c3f7').
                    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    random_color = f'#{r:02x}{g:02x}{b:02x}'
                
                # Assign the new unique color to the category.
                category_to_color[category] = random_color
                # Add the new color to the set of used colors.
                used_colors.add(random_color)

        # --- Plotting and Annotation ---
        # Set up the figure size for the plot.
        plt.figure(figsize=(5, 5))

        # Plot points for each category to associate them with the correct legend entry.
        for category in unique_categories:
            # Find the indices of words belonging to the current category.
            indices = [i for i, word in enumerate(labels) if word_to_category[word] == category]
            # Get the coordinates for the current category using the indices.
            category_coords = coords[indices]
            
            # Create a scatter plot for the current category's points.
            plt.scatter(
                category_coords[:, 0],
                category_coords[:, 1],
                color=category_to_color[category],
                s=120,
                alpha=0.9,
                label=category
            )

        # Add text annotations for each individual data point.
        for i, word in enumerate(labels):
            plt.annotate(word, (coords[i, 0], coords[i, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10)

        # --- Final Plot Adjustments ---
        # Set the plot title and axis labels.
        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        # Add a grid for better readability.
        plt.grid(True, alpha=0.3)
        # Display the legend to show category-color mappings.
        plt.legend(loc='lower left', title="Categories")
        # Adjust plot to ensure everything fits without overlapping.
        plt.tight_layout()
        # Display the final visualization.
        plt.show()

    except KeyError as e:
        # Handle cases where a label is not found in the categorization dictionary.
        print(f"Error: Word {e} from the label list was not found in the dictionary.")
        print("Please ensure that the label list and the categorization dictionary are consistent.")

# test code
# a = ['which', 'team', 'is', 'the', '``', 'champion', "''", 'of', 'the', 'world', 'cup', '2026', '.', '❤️', 'espana', '.']
# for cont, word in get_sliding_context(a, 3):
#     print(f"context: {cont}  center: {word}")

def filter_recipe_dataset(input_path, output_path="recipes_fruit_veg.csv"):
    """
    Filters the raw Food.com recipe dataset to create a smaller subset
    containing only mutually exclusive fruit or vegetable recipes.

    This function reads a large recipe dataset, categorizes each recipe
    based on keywords in its ingredients, and filters it to keep only recipes
    that are exclusively fruit-based or exclusively vegetable-based. The
    resulting subset is then saved to a new CSV file.

    Args:
        input_path: The file path for the original recipe dataset CSV file.
        output_path: The file path where the filtered CSV will be saved.
    """
    print(f"Loading the raw dataset from '{input_path}'...")
    # Read the dataset from the specified path, with error handling for missing files.
    try:
        df = pd.read_csv(input_path)
        # return df
    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_path}'")
        return

    # Define keywords for categorization.
    fruit_keywords = [
        "apple", "banana", "orange", "strawberry", "grape", "mango",
        "pineapple", "peach", "pear", "cherry", "berry", "lemon",
        "lime", "melon",
    ]
    vegetable_keywords = [
        "carrot", "broccoli", "spinach", "potato", "tomato", "onion",
        "garlic", "pepper", "lettuce", "cucumber", "celery", "mushroom",
        "corn", "bean", "pea", "cabbage", "asparagus",
    ]

    def categorize_recipe(ingredients_str):
        """Categorizes a recipe as 'fruit', 'vegetable', or 'other'."""
        try:
            # Safely parse the string representation of the ingredient list.
            ingredients_list = ast.literal_eval(ingredients_str)
            ingredients_text = " ".join(ingredients_list).lower()

            # Check for the presence of fruit or vegetable keywords.
            has_fruit = any(keyword in ingredients_text for keyword in fruit_keywords)
            has_veg = any(keyword in ingredients_text for keyword in vegetable_keywords)

            # Assign mutually exclusive categories.
            if has_fruit and not has_veg:
                return "fruit"
            if has_veg and not has_fruit:
                return "vegetable"
            
            # Return 'other' for recipes with both or no relevant keywords.
            return "other"
        
        except (ValueError, SyntaxError):
            # Handle potential parsing errors for malformed ingredient strings.
            return "other"

    print("Categorizing recipes based on ingredient keywords...")
    # Apply the categorization function to each row in the DataFrame.
    df["category"] = df["ingredients"].apply(categorize_recipe)

    # Filter the DataFrame to keep only 'fruit' and 'vegetable' categories.
    filtered_df = df[df["category"].isin(["fruit", "vegetable"])].copy()

    # Define the specific columns to keep in the final dataset.
    columns_to_keep = ["name", "id", "minutes", "ingredients", "steps", "category"]
    subset_df = filtered_df[columns_to_keep]

    print("Filtering complete.")
    print(f"Found {len(subset_df[subset_df['category'] == 'fruit'])} fruit recipes.")
    print(f"Found {len(subset_df[subset_df['category'] == 'vegetable'])} vegetable recipes.")

    print(f"\nSaving the subset data to '{output_path}'...")
    # Save the final filtered DataFrame to a CSV file.
    subset_df.to_csv(output_path, index=False)

    print(f"Success! Subset dataset saved to '{output_path}'.")