from flask import Flask
import os
import logging
from joblib import dump, load
from flask import request, jsonify, abort
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import json


from question_tagger import QuestionTagger
import torch
import numpy as np
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

client = WebClient(token=os.environ.get("xoxb-2303935478769-2297880316244-vKtkyqSkDMB6Ek6qeIGEJWtl"))
logger = logging.getLogger(__name__)
app = Flask(__name__)
model = load("naivebayes.joblib")

BERT_MODEL_NAME = 'bert-base-cased'
bert = QuestionTagger(n_classes=3)
bert.load_state_dict(torch.load("trained_model.pth"))

@app.route("/", methods=["POST"])
def action():

	payload = json.loads(request.form["payload"])
	actions = payload["actions"][0]
	action_id = actions["action_id"]

	if action_id == "post":
		value = json.loads(actions["value"])
		question = value["question"]
		channel_id = value["channel_id"]

		try:
			return client.chat_postMessage(
				token = "xoxb-2303935478769-2297880316244-vKtkyqSkDMB6Ek6qeIGEJWtl",
				channel=channel_id,
				text=question
			)

		except SlackApiError as e:
			print(f"Error: {e}")


@app.route('/where', methods=['POST'])
def recommend():
    # Parse the parameters you need
	text = request.form.get('text', None)

	if text == None or text == '':
		return "Please enter a question."

	#channel = model.predict([text])[0]
	channel = predict(text)
	channel_id = findChannelId(channel)
	val = { "question" : text, "channel_id" : channel_id}

	return jsonify({
			"blocks": [
				{
					"type": "section",
					"text": {
						"type": "mrkdwn",
						"text": "*"+text+"*"
					}
				},
				{
					"type": "section",
					"text": {
						"type": "mrkdwn",
						"text": "I suggest you post your question in <#"+channel_id+"|"+channel+">."
					}
				},
				{
					"type": "actions",
					"block_id": "actionblock789",
					"elements": [
						{
							"type": "button",
							"text": {
								"type": "plain_text",
								"text": "Post Question"
							},
							"style": "primary",
							"value": json.dumps(val),
							"action_id": "post"
						}
					]
				}
			]
		})

def predict(text):
	test_comment = text
	tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
	encoding = tokenizer.encode_plus(
	  test_comment,
	  add_special_tokens=True,
	  max_length=128,
	  return_token_type_ids=False,
	  padding="max_length",
	  return_attention_mask=True,
	  return_tensors='pt',
	)

	_, test_prediction = bert(encoding["input_ids"], encoding["attention_mask"])
	test_prediction = test_prediction.flatten().detach().numpy()
	LABEL_COLUMNS = ['cloud', 'node', 'python']
	return LABEL_COLUMNS[np.argmax(test_prediction)]

def findChannelId(channel_name):
    try:
        # Call the conversations.list method using the WebClient
        for result in client.conversations_list(token = "xoxb-2303935478769-2297880316244-vKtkyqSkDMB6Ek6qeIGEJWtl"):
            for channel in result["channels"]:
                if channel["name"] == channel_name:
                    return channel["id"]

    except SlackApiError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)