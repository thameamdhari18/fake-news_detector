

---

**Fake News Detection**

This project aims to detect fake news using a Logistic Regression model trained on a dataset of true and fake news articles. The dataset is preprocessed, vectorized using TF-IDF, and then used to train the model. The project also includes a Flask API for predicting the likelihood of a news article being fake.

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [API](#api)
- [Evaluation](#evaluation)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

**Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/ThameamDhari/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the `archive.zip` file containing the datasets in the project directory.

**Usage**

1. Extract the datasets from the `archive.zip` file:
    ```python
    import zipfile

    archive_path = 'C:\\Users\\thame\\Downloads\\archive.zip'
    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extract('True.csv')
        archive.extract('Fake.csv')
    ```

2. Run the script to preprocess the data, train the model, and start the Flask app:
    ```bash
    python app.py
    ```

3. Use the API to predict whether a news article is fake:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"text": "Your news article text here"}' http://127.0.0.1:5000/predict
    ```

**Project Structure**

```
fake-news-detection/
│
├── True.csv                   # True news dataset
├── Fake.csv                   # Fake news dataset
├── app.py                     # Main application file
├── requirements.txt           # Required packages
├── fake_news_model.pkl        # Trained model
├── vectorizer.pkl             # TF-IDF vectorizer
└── README.md                  # Project README file
```

**Dataset**

The dataset contains two CSV files:
- `True.csv`: Contains true news articles.
- `Fake.csv`: Contains fake news articles.

Each file has the following columns:
- `title`: The title of the news article.
- `text`: The content of the news article.
- `subject`: The subject of the news article.
- `date`: The date the news article was published.

**Model**

The Logistic Regression model is trained on the news articles after preprocessing and vectorization using TF-IDF. The preprocessing steps include converting the text to lowercase, removing punctuation, and removing numbers.

**API**

The Flask API provides an endpoint for predicting whether a news article is fake:

- `POST /predict`: Predict whether the provided news article text is fake.
  - Request body: `{"text": "Your news article text here"}`
  - Response: `{"prediction": 0}` (0 for fake, 1 for true)

**Evaluation**

The model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

**Output**

After running the script, you should see output similar to the following:

```
Combined Data:
                                               title  subject               date                                               text  label
0  As U.S. budget fight looms, Republicans flip f...  politicsNews  December 31, 2017  WASHINGTON (Reuters) - The head of a conservativ...      1
1  U.S. military to accept transgender recruits o...  politicsNews  January 1, 2018  WASHINGTON (Reuters) - Transgender people will b...      1
2  Senate Democrats question Trump's nuclear auth...  politicsNews  December 31, 2017  WASHINGTON (Reuters) - Two Senate Democrats hav...      1
3  FBI Russia probe helped by Australian diplomat...  politicsNews  December 30, 2017  WASHINGTON (Reuters) - Australian officials rel...      1
4  Trump wants voters focused on tax reform, not ...  politicsNews  December 31, 2017  WEST PALM BEACH, Fla. (Reuters) - U.S. Presiden...      1

Accuracy: 0.9873271889400922
Precision: 0.9827709978463748
Recall: 0.9928425357873211
F1-score: 0.9877810361681334
```

**Contributing**

Contributions are welcome! Please fork the repository and create a pull request with your changes.

 **License**

This project is licensed under the MIT License.

---
