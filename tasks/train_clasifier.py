import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_hist(df):
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Combine all features into a single DataFrame
    data_to_plot = df[['Smalles_angle', 'Sum_of_angles', 'Area_diff']]

    # Create a KDE plot for all features combined
    sns.kdeplot(data=data_to_plot, shade=True)

    # Set labels and title
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.title('Combined Probability Density Function')

    plt.tight_layout()
    plt.show()


# Load your DataFrame
path = r'C:\Users\IlyaY\Desktop\לימודים\תשפג\ק\עיבוד תמונה\Lapis\all\data.csv'
df = pd.read_csv(path)  # Replace 'your_data.csv' with your DataFrame file


# Extract columns
X = df[['Smalles_angle', 'Sum_of_angles', 'Area_diff']]  # Replace with your parameter columns
y = df['Label']  # Replace with your true label column

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (e.g., Logistic Regression) with multiple features
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Generate predicted probabilities on the test set with multiple features
y_probs = classifier.predict_proba(X_test)[:, 1]  # Assuming class 1 is the positive class

# Calculate the F1-score for various threshold values
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # You can adjust these thresholds
f1_scores = []

for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# Find the threshold that maximizes the F1-score for the multi-feature model
optimal_threshold = thresholds[f1_scores.index(max(f1_scores))]

# Save the trained multi-feature classifier to a file
# joblib.dump(classifier, r"C:\Users\IlyaY\Desktop\לימודים\תשפג\ק\עיבוד תמונה\Lapis\all\trained_classifier.pkl")

# Convert predicted probabilities to binary predictions using the threshold
y_pred = (y_probs >= optimal_threshold).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract TP, TN, FP, FN from the confusion matrix
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

# Create a DataFrame to display the results as a table
result_df = pd.DataFrame({'Metric': ['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)'],
                          'Count': [TP, TN, FP, FN]})

# Display the DataFrame as a table in Jupyter Notebook
print(result_df)