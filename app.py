from flask import Flask, render_template, request
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import torch.nn as nn

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = 'pvqa/'

pvqa_dic_path = base_path + 'pvqa_dictionary.pkl'
test_img_id2idx_path = base_path + 'test_img_id2idx.pkl'
train_img_id2idx_path = base_path + 'train_img_id2idx.pkl'
val_img_id2idx_path = base_path + 'val_img_id2idx.pkl'

pvqa_dic_pkl = pd.read_pickle(pvqa_dic_path)
test_img_id2idx_pkl = pd.read_pickle(test_img_id2idx_path)
train_img_id2idx_pkl = pd.read_pickle(train_img_id2idx_path)
val_img_id2idx_pkl = pd.read_pickle(val_img_id2idx_path)

qas_base_path = base_path + 'qas/'

q2a_path = qas_base_path + 'q2a.pkl'
ans2lbl_path = qas_base_path + 'ans2label.pkl'
qid2a_path = qas_base_path + 'qid2a.pkl'
qid2q_path = qas_base_path + 'qid2q.pkl'
test_vqa_path = qas_base_path + 'test_vqa.pkl'
train_vqa_path = qas_base_path + 'train_vqa.pkl'
val_vqa_path = qas_base_path + 'val_vqa.pkl'
train_qa_path = qas_base_path + 'train/train_qa.pkl'
trainval_ans2lbl_path = qas_base_path + 'trainval_ans2label.pkl'
trainval_lbl2ans_path = qas_base_path + 'trainval_label2ans.pkl'

q2a_pairs = pd.read_pickle(q2a_path)
ans2lbl_pairs = pd.read_pickle(ans2lbl_path)
qid2a_pairs = pd.read_pickle(qid2a_path)
qid2q_pairs = pd.read_pickle(qid2q_path)
test_vqa_pairs = pd.read_pickle(test_vqa_path)
train_vqa_pairs = pd.read_pickle(train_vqa_path)
train_qa_pairs = pd.read_pickle(train_qa_path)
val_vqa_pairs = pd.read_pickle(val_vqa_path)
trainval_lbl2ans_pairs = pd.read_pickle(trainval_lbl2ans_path)
trainval_ans2lbl_pairs = pd.read_pickle(trainval_ans2lbl_path)

train_vqa_df = pd.DataFrame(train_vqa_pairs)
val_vqa_df = pd.DataFrame(val_vqa_pairs)
test_vqa_df = pd.DataFrame(test_vqa_pairs)

tr_q = list(train_vqa_df['sent'])
val_q = list(val_vqa_df['sent'])
ts_q = list(test_vqa_df['sent'])

tr_a = [q2a_pairs[each] for each in tr_q]
val_a = [q2a_pairs[each] for each in val_q]
ts_a = [q2a_pairs[each] for each in ts_q]

qs = [tr_q, val_q, ts_q]
ans = [tr_a, val_a, ts_a]

del tr_q, tr_a, ts_q, ts_a, val_q, val_a

val_lens = [len(each) for each in qs[1]]
tr_lens = [len(each) for each in qs[0]]
ts_lens = [len(each) for each in qs[2]]

del tr_lens, val_lens, ts_lens

idx = 0
ans2lbl_len = 4092
ans2lbl_keys = list(ans2lbl_pairs.keys())

for each in ans:
	for i in range(len(each)):
		if each[i] not in ans2lbl_keys:
			ans2lbl_pairs[each[i]] = ans2lbl_len + idx
			idx += 1

del ans2lbl_keys

num_classes = len(list(ans2lbl_pairs.keys()))
anss = list(ans2lbl_pairs.keys())
lbls = list(range(0, len(anss)))
ans2lbl_pairs = dict(zip(anss, lbls))

del anss, lbls


dfs = [train_vqa_df, val_vqa_df, test_vqa_df]

del train_vqa_df, val_vqa_df, test_vqa_df

for i in range(len(dfs)):
    dfs[i]['question'] = dfs[i]['sent']
    dfs[i]['answer'] = pd.Series(ans[i])
    dfs[i]['label'] = pd.Series([ans2lbl_pairs[each] for each in ans[i]])

    if i == 0:
        dfs[i]['img_id'] = base_path + 'images/train/' + dfs[i]['img_id']
    elif i == 1:
        dfs[i]['img_id'] = base_path + 'images/val/' + dfs[i]['img_id']
    else:
        dfs[i]['img_id'] = base_path + 'images/test/' + dfs[i]['img_id']

    dfs[i].drop(['sent', 'answer_type', 'question_id', 'question_type'], axis=1, inplace=True)

def extract_free_form(df):
    # Filter out the rows where answer is not yes, not no, and not a number
    return df[(df['answer'] != 'yes') & (df['answer'] != 'no')]

### Give me a function that extracts all the rows from the datafram having yes or no as the answer
def extract_yes_no(df):
    return df[(df['answer'] == 'yes') | (df['answer'] == 'no')]

# Load your dataframes
val_yes_no = extract_yes_no(dfs[1])
val_free_form = extract_free_form(dfs[1])

# Initialize BERT tokenizer
text_processor = BertTokenizer.from_pretrained('bert-base-uncased')

# Define transformations for images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to create label mappings dynamically
# Function to create label mappings dynamically
def create_label_mappings(dataframe):
    label_to_answer = {}
    for index, row in dataframe.iterrows():
        label_to_answer[row['label']] = row['answer']
    return label_to_answer

# Create label mappings for the test dataset
label_mappings = create_label_mappings(dfs[1])

# VQAModel class definition
class Attention(nn.Module):
    def __init__(self, image_dim, question_dim, hidden_dim):
        super(Attention, self).__init__()
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        self.question_fc = nn.Linear(question_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, image_features, question_features):
        # Linear transformations
        image_proj = self.image_fc(image_features)
        question_proj = self.question_fc(question_features).unsqueeze(1)
        
        # Combine projections
        combined_proj = torch.tanh(image_proj + question_proj)
        attention_weights = torch.softmax(self.fc(combined_proj), dim=1)
        
        # Weighted sum of image features
        attended_image_features = (attention_weights * image_features).sum(dim=1)
        
        return attended_image_features, attention_weights

class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel, self).__init__()
        # Image feature extractor using VGG-19
        vgg = models.vgg19(pretrained=True)
        self.vgg_features = vgg.features
        self.vgg_classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.vgg_fc = nn.Linear(4096, 512)

        # Question feature extractor using BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_fc = nn.Linear(768, 512)

        # Attention layers
        self.attention1 = Attention(512, 512, 512)
        self.attention2 = Attention(512, 512, 512)

        # Combined classifier
        self.fc1 = nn.Linear(512 + 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_answers)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, images, input_ids, attention_mask):
        # Image features
        with torch.no_grad():
            img_features = self.vgg_features(images)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.vgg_classifier(img_features)
        img_features = self.vgg_fc(img_features)
        
        # Question features
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        ques_features = self.bert_fc(bert_output.pooler_output)

        # Attention layers
        attended_img_features1, _ = self.attention1(img_features, ques_features)
        attended_img_features2, _ = self.attention2(attended_img_features1, ques_features)

        # Concatenate features
        combined_features = torch.cat((attended_img_features2, ques_features), dim=1)

        # Classification
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc3(x)

        return output

model_path_free_form = 'vqa_model_free_form.pth'
model_path_yes_no = 'vqa_model_yes_no.pth'

# Function to generate inference
def generate_inference(question, image_path, model, label_mappings):
    image = Image.open(image_path).convert("RGB")
    image_transform = data_transforms(image)
    image_tensor = image_transform.unsqueeze(0)

    encoding = text_processor(
        question,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)

    with torch.no_grad():
        outputs = model(image_tensor.to(device), input_ids.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device))
        _, predicted = torch.max(outputs.data, 1)
    
    predicted_answer = label_mappings.get(predicted.item(), "Unknown")

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return predicted_answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_choice = request.form['model_choice']
        index = int(request.form['index'])

        if model_choice == 'Yes / No':
            # Load your models (adjust paths as needed)
            model = VQAModel(num_classes)
            model = model.to(device)
            model.load_state_dict(torch.load(model_path_yes_no))
            model.eval()
            inference_df = val_yes_no
        elif model_choice == 'Free Form':
            # Load your models (adjust paths as needed)
            model = VQAModel(num_classes)
            model = model.to(device)
            model.load_state_dict(torch.load(model_path_free_form))
            model.eval()
            inference_df = val_free_form

        question = inference_df.iloc[index]['question']
        image_path = inference_df.iloc[index]['img_id'] + '.jpg'
        actual_answer = inference_df.iloc[index]['answer']

        prediction = generate_inference(question, image_path, model, label_mappings)

        return render_template('result.html', model_choice=model_choice, index=index, question=question, actual_answer=actual_answer, prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
