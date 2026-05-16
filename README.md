# News Classification

This project delivers a end-to-end pipeline for training a News Classifier, that can label news article into  different categories. 

## About Data

I have used a open source dataset from kaggle [click here for link](https://www.kaggle.com/datasets/rmisra/news-category-dataset), that has 200k+ rows of data of news articles including their headline, link, short description and the category of that news article.

### *Data Featuring*

* **Class Imbalance**: The Data had class imbalance, i.e. some classes were much common while some were rare. This is actually a real world scenerio, as news on *POLITICS* is much more common than news on *EDUCATION*. This would however make our model biased towards these categories, so to fix that I used class weights to penalize the error on rare categories more.
  
* **Overlapping Categories**: Some categories were very similar or actually same (*like WORLDPOST and THE WORLDPOST*). This creates confussion for the model to categorise them and dropped the accuracy score. To fix this issue I merged the categories that were similar or vague.
  
* **Feature Selection**: The Dataset had columns for data of article, link of the article which I don't need for the training purpose. I only need the text part of the news to  train the NLP model. So, I merged the columns that have text part of the data and dropped the other columns. That left me with two columns one containing text and other containing label.


## About the model

I have used a pre-trained `BERT` model from hugging face's transformers library and fine tuned it for classification task. 

*Why BERT model?*

* BERT (Bidirectional Encoder Representations from Transformers), is a model trained by google on entire English Wikipedia and a large collection of books. It has 110 million parameters. The model has a deep understanding of english grammer, word relationships, context, and semantics.

* Instead of training a neural network  from scratch I leveraged  this capability of the BERT and then add a classification head to make it predict class probabilities of various class and then get a label.

## Training Process

I used a two step approach - first training on full data with imbalanced classes and get a baseline which has a general intuition of the class imbalance in the real world. Then I fine-tuned it on a balanced subset of the full data.

### *Initial Training*:

Trained the model on the full dataset and got the following results.

* Accuracy on the validation set : 0.6711
* Accuracy on the test set : 0.6704
* Classification Report:
  Test Accuracy: 0.6704

Classification Report:

    Labels          precision    recall  f1-score   support

    ARTS            0.57      0.66      0.61       784
    BLACK VOICES    0.47      0.61      0.53       917
    BUSINESS        0.54      0.56      0.55      1198
    COLLEGE         0.39      0.63      0.48       229
    COMEDY          0.52      0.61      0.56      1080
    CRIME           0.59      0.64      0.62       712
    DIVORCE         0.76      0.87      0.81       685
    EDUCATION       0.27      0.67      0.39       203
    ENTERTAINMENT   0.81      0.66      0.73      3473
    ENVIRONMENT     0.49      0.52      0.51       289
    FIFTY           0.32      0.56      0.41       280
    FOOD & DRINK    0.78      0.69      0.73      1268
    GOOD NEWS       0.38      0.49      0.43       280
    GREEN           0.38      0.57      0.46       524
    HEALTHY LIVING  0.45      0.48      0.47      1339
    HOME & LIVING   0.80      0.87      0.83       864
    IMPACT          0.36      0.49      0.41       697
    LATINO VOICES   0.41      0.69      0.52       226
    MEDIA           0.41      0.70      0.52       589
    MONEY           0.49      0.70      0.58       351
    PARENTS         0.81      0.70      0.75      2549
    POLITICS        0.93      0.59      0.72      7121
    QUEER VOICES    0.79      0.74      0.77      1270
    RELIGION        0.56      0.74      0.64       515
    SCIENCE         0.53      0.64      0.58       441
    SPORTS          0.74      0.81      0.78      1015
    STYLE & BEAUTY  0.86      0.84      0.85      2414
    TASTE           0.39      0.66      0.49       419
    TECH            0.50      0.67      0.57       421
    TRAVEL          0.84      0.85      0.84      1980
    U.S. NEWS       0.22      0.60      0.32       275
    WEDDINGS        0.85      0.84      0.84       731
    WEIRD NEWS      0.46      0.49      0.48       555
    WELLNESS        0.82      0.65      0.73      3589
    WOMEN           0.37      0.58      0.46       714
    WORLD NEWS      0.76      0.73      0.74      1909

    accuracy                           0.67     41906
    macro avg       0.57      0.66      0.60     41906
    weighted avg    0.72      0.67      0.68     41906

### *Fine Tuning*:

I trained the model on 1000 samples of each class to see if that improves the performance.

I got the following results:

* Accuracy on validation set of balanced data: 0.7987
* Accuracy on test set after fine tuning: 0.6568
  
Classification Report:

    Label            precision    recall  f1-score   support

    ARTS                0.49      0.69      0.57       784
    BLACK VOICES        0.47      0.61      0.53       917
    BUSINESS            0.55      0.53      0.54      1198
    COLLEGE             0.38      0.74      0.51       229
    COMEDY              0.52      0.61      0.56      1080
    CRIME               0.54      0.70      0.61       712
    DIVORCE             0.78      0.84      0.81       685
    EDUCATION           0.33      0.82      0.47       203
    ENTERTAINMENT       0.84      0.61      0.70      3473
    ENVIRONMENT         0.41      0.58      0.48       289
    FIFTY               0.24      0.64      0.35       280
    FOOD & DRINK        0.75      0.73      0.74      1268
    GOOD NEWS           0.37      0.49      0.42       280
    GREEN               0.37      0.53      0.44       524
    HEALTHY LIVING      0.45      0.43      0.44      1339
    HOME & LIVING       0.80      0.86      0.83       864
    IMPACT              0.41      0.45      0.43       697
    LATINO VOICES       0.39      0.74      0.51       226
    MEDIA               0.41      0.68      0.51       589
    MONEY               0.46      0.72      0.56       351
    PARENTS             0.81      0.71      0.76      2549
    POLITICS            0.93      0.55      0.69      7121
    QUEER VOICES        0.78      0.74      0.76      1270
    RELIGION            0.53      0.72      0.61       515
    SCIENCE             0.51      0.66      0.57       441
    SPORTS              0.71      0.83      0.77      1015
    STYLE & BEAUTY      0.88      0.81      0.84      2414
    TASTE               0.39      0.68      0.50       419
    TECH                0.46      0.70      0.56       421
    TRAVEL              0.86      0.81      0.83      1980
    U.S. NEWS           0.18      0.57      0.28       275
    WEDDINGS            0.81      0.88      0.84       731
    WEIRD NEWS          0.41      0.52      0.46       555
    WELLNESS            0.82      0.63      0.71      3589
    WOMEN               0.40      0.56      0.47       714
    WORLD NEWS          0.72      0.74      0.73      1909

    accuracy                                0.66     41906
    macro avg           0.56      0.67      0.59     41906
    weighted avg        0.72      0.66      0.67     41906

Fine-tuning on balanced data marginally improved recall on rare classes but slightly hurt overall accuracy due to the model losing its calibration to the natural class distribution. This indicates that the model has learned the important relationship about data in its initial training and fine-tuning didn't help much in making them better.

## How to run this model?

### Requirements

Python 3.10+

Install dependencies:
pip install torch transformers scikit-learn pandas numpy tqdm

GPU recommended for training. CPU will work for inference only.

### Dataset

Download the dataset from Kaggle:
[click here](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

Place `News_Category_Dataset_v3.json` in the project root directory.

### Training

Open `news-classifier.ipynb` and run all cells in order.
Training time on the full dataset will depend on your GPU.
The best model is saved automatically to `news_model_best/`.

### Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import torch, pickle

model = AutoModelForSequenceClassification.from_pretrained("news_model_best")
tokenizer = AutoTokenizer.from_pretrained("news_model_best")

with open("news_model_best/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

predict("NASA launches new telescope to explore deep space")
# → 'SCIENCE'
```
## Limitations
- Categories with vague definitions (FIFTY, GOOD NEWS, IMPACT) consistently 
  underperform regardless of training approach
- Politics recall is lower than expected due to semantic overlap with U.S. NEWS, 
  WORLD NEWS, and WOMEN categories
- Model assumes English input only