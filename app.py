import os
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set Matplotlib backend to 'Agg' for non-interactive plotting
import matplotlib.pyplot as plt
import google.generativeai as genai
from flask import Flask, request, render_template, session, redirect, url_for
from dotenv import load_dotenv, find_dotenv
import uuid # Import uuid for unique IDs
import sys # Import sys for stdout redirection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)
app.secret_key = os.urandom(24) # Secret key for session management

# Global dictionary to store Dataframes
dataframes = {}
# Global dictionary to store visualizations
visualizations = {}

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key or gemini_api_key == "YOUR_API_KEY":
    raise ValueError("GEMINI_API_KEY not found or not set in .env file. Please set your actual Gemini API key.")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_df_summary(df):
    """Generates an HTML summary of the DataFrame."""
    # Manually construct the info string to avoid issues with df.info() output capture
    info_str = "Data types:\n"
    for col, dtype in df.dtypes.items():
        info_str += f"{col}: {dtype}\n"
    info_str += f"\nShape: {df.shape}\n"
    info_str += f"Number of entries: {len(df)}\n"

    return df.head().to_html() + \
           f"<h3>Columns and Data Types:</h3><pre>{info_str}</pre>"

@app.route('/', methods=['GET', 'POST'])
def index():
    df_summary = None
    visualization = None
    error_message = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                error_message = "No selected file"
            elif file:
                try:
                    df = pd.read_csv(file)
                    df_id = str(uuid.uuid4())
                    dataframes[df_id] = df # Store DataFrame in global dictionary
                    session['df_id'] = df_id # Store ID in session
                    df_summary = get_df_summary(df)
                except Exception as e:
                    error_message = f"Error reading CSV: {e}"
        elif 'query' in request.form:
            user_query = request.form['query']
            if 'df_id' in session and session['df_id'] in dataframes:
                df = dataframes[session['df_id']]
                try:
                    # Generate Python code using Gemini
                    prompt = f"""
                    You are a Python data analysis assistant.
                    The user has uploaded a pandas DataFrame.
                    The DataFrame has the following columns: {df.columns.tolist()}
                    The user's question is: "{user_query}"

                    Generate ONLY Python code using pandas and matplotlib to answer the question and create a visualization.
                    DO NOT include any conversational text, explanations, or markdown code fences (```python).
                    The code should:
                    1. Assume the DataFrame is named `df`.
                    2. Use `matplotlib.pyplot` for plotting.
                    3. Save the plot to a BytesIO object as a PNG image.
                    4. NOT include `plt.show()`.
                    5. NOT include `import pandas as pd` or `import matplotlib.pyplot as plt` or `import io` or `import base64` or `import sys` or `import uuid`.
                    6. NOT include any print statements for the plot.
                    7. Ensure the plot is clear and readable.
                    8. If the query asks for a specific type of plot (e.g., "bar chart", "scatter plot"), use that. Otherwise, choose an appropriate plot type.
                    9. If the query asks for a summary or specific data points, provide code to calculate and print them.
                    10. If the query asks for a visualization, ensure the code generates a plot.
                    11. If the query asks to clear the visualization, respond with "CLEAR_VISUALIZATION".

                    Example of expected output for a visualization query (NO ```python fences):
                    plt.figure(figsize=(10, 6))
                    df['column'].value_counts().plot(kind='bar')
                    plt.title('Title of Plot')
                    plt.xlabel('X-axis Label')
                    plt.ylabel('Y-axis Label')
                    plt.tight_layout()
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png')
                    img_buffer.seek(0)

                    Example of expected output for a data summary query (NO ```python fences):
                    print(df['column'].describe())
                    """
                    response = model.generate_content(prompt)
                    generated_code = response.text.strip()
                    # Remove markdown code fences if Gemini still includes them
                    if generated_code.startswith("```python"):
                        generated_code = generated_code.lstrip("```python").rstrip("```").strip()

                    if generated_code == "CLEAR_VISUALIZATION":
                        session.pop('visualization_id', None) # Clear visualization ID from session
                        visualizations.pop(session.get('visualization_id'), None) # Remove visualization from global dict
                        session.pop('query_result_text', None)
                        return redirect(url_for('index'))

                    # Securely execute the generated code
                    exec_globals = {'df': df, 'plt': plt, 'io': io, 'base64': base64, 'pd': pd, 'StringIO': io.StringIO}
                    exec_locals = {'img_buffer': None, 'query_result_text': None}

                    # Redirect stdout to capture print statements
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()

                    try:
                        exec(generated_code, exec_globals, exec_locals)
                        query_result_text = sys.stdout.getvalue()

                        if exec_locals['img_buffer']:
                            img_base64 = base64.b64encode(exec_locals['img_buffer'].getvalue()).decode('utf-8')
                            viz_id = str(uuid.uuid4())
                            visualizations[viz_id] = img_base64 # Store visualization in global dictionary
                            session['visualization_id'] = viz_id # Store ID in session
                            session.pop('query_result_text', None) # Clear text if visualization is present
                        elif query_result_text:
                            session['query_result_text'] = query_result_text
                            session.pop('visualization_id', None) # Clear visualization ID if text is present
                            visualizations.pop(session.get('visualization_id'), None) # Remove visualization from global dict
                        else:
                            error_message = "Gemini generated code but no visualization or text output was produced."

                    except Exception as e:
                        error_message = f"Error executing generated code: {e}"
                        print(f"Generated code that caused error:\n{generated_code}") # For debugging
                    finally:
                        sys.stdout = old_stdout # Restore stdout
                        plt.close('all') # Close all plots to free memory

                except Exception as e:
                    error_message = f"Error with Gemini API: {e}"
            else:
                error_message = "Please upload a CSV file first."
        elif 'clear_visualization' in request.form:
            session.pop('visualization_id', None) # Clear visualization ID from session
            visualizations.pop(session.get('visualization_id'), None) # Remove visualization from global dict
            session.pop('query_result_text', None)
            return redirect(url_for('index'))

    if 'df_id' in session and session['df_id'] in dataframes:
        df = dataframes[session['df_id']]
        df_summary = get_df_summary(df)
    
    # Retrieve visualization from global dictionary using ID from session
    visualization_id = session.get('visualization_id', None)
    visualization = visualizations.get(visualization_id, None) if visualization_id else None
    query_result_text = session.get('query_result_text', None)

    return render_template('index.html', df_summary=df_summary, visualization=visualization, error_message=error_message, query_result_text=query_result_text)

@app.route('/clusters', methods=['GET'])
def clusters():
    visualization = None
    error_message = None

    if 'df_id' in session and session['df_id'] in dataframes:
        df = dataframes[session['df_id']]
        
        try:
            # Select only numerical columns for clustering
            numerical_df = df.select_dtypes(include=['number'])
            
            if numerical_df.empty:
                error_message = "No numerical columns found for clustering."
            else:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numerical_df)

                # Apply PCA to reduce to 2 dimensions
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_data)

                # Apply K-Means clustering
                # Using a fixed number of clusters (e.g., 3) for simplicity
                # In a real application, you might use elbow method or silhouette score to determine optimal k
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # n_init to suppress warning
                cluster_labels = kmeans.fit_predict(pca_data)

                # Calculate feature contributions to PCA components
                feature_names = numerical_df.columns
                pca_components = pca.components_ # This holds the loadings

                pca_features_contribution = []
                for i, component in enumerate(pca_components):
                    # Create a list of (feature_name, absolute_loading) tuples
                    loadings = [(feature_names[j], abs(loading)) for j, loading in enumerate(component)]
                    # Sort by absolute loading in descending order
                    loadings.sort(key=lambda x: x[1], reverse=True)
                    # Get top 3
                    top_features = loadings[:3]
                    pca_features_contribution.append({
                        'component': i + 1,
                        'features': [{'name': name, 'weight': weight} for name, weight in top_features]
                    })
                session['pca_features_contribution'] = pca_features_contribution # Store in session

                # Calculate mean values of top 3 PCA features for each cluster
                # First, get the names of the top 3 features across all components
                top_3_feature_names = []
                for comp_data in pca_features_contribution:
                    for feature in comp_data['features']:
                        if feature['name'] not in top_3_feature_names:
                            top_3_feature_names.append(feature['name'])
                top_3_feature_names = top_3_feature_names[:3] # Ensure only top 3 unique features

                if top_3_feature_names:
                    # Add cluster labels to the numerical DataFrame
                    numerical_df_with_clusters = numerical_df.copy()
                    numerical_df_with_clusters['cluster'] = cluster_labels

                    # Calculate mean values for top features by cluster
                    cluster_means_df = numerical_df_with_clusters.groupby('cluster')[top_3_feature_names].mean()
                    
                    # Convert to a list of dictionaries for easier Jinja2 templating
                    cluster_means_data = []
                    for cluster_id, row in cluster_means_df.iterrows():
                        cluster_data = {'cluster': cluster_id}
                        for feature_name in top_3_feature_names:
                            cluster_data[feature_name] = row[feature_name]
                        cluster_means_data.append(cluster_data)
                    session['cluster_means'] = cluster_means_data # Store in session
                    session['cluster_mean_features'] = top_3_feature_names # Store feature names for header

                # Create scatter plot
                plt.figure(figsize=(10, 7))
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
                plt.title('K-Means Clusters (PCA Reduced)')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.colorbar(scatter, label='Cluster Label')
                plt.grid(True)
                plt.tight_layout()

                # Save plot to BytesIO object
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                plt.close('all') # Close plot to free memory

                # Store visualization in global dictionary
                viz_id = str(uuid.uuid4())
                visualizations[viz_id] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                session['visualization_id'] = viz_id # Store ID in session
                session.pop('query_result_text', None) # Clear text if visualization is present

        except Exception as e:
            error_message = f"Error performing clustering: {e}"
            plt.close('all') # Ensure plot is closed even on error
    else:
        error_message = "Please upload a CSV file first to perform clustering."
    
    # Redirect to index page to display the visualization
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
