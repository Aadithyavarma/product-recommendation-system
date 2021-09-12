from flask import Flask,render_template,url_for,request

from model import get_product_recommendation


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    actual_username = request.form["username"]
    username, top_5_items = get_product_recommendation(actual_username)
    print(top_5_items)
    if not username:
        return render_template('index.html', error_message=top_5_items)
    else:
        return render_template('index.html', final_product_list=top_5_items, username=username)


if __name__ == '__main__':
	app.run(debug=True)