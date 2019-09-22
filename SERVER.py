from flask import Flask, render_template,url_for,request
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from spacy.lang.en import English
import spacy
import re
import sqlite3 as sql	

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/list')
def list():

   #CONNECTING TO  DATASET
   con = sql.connect("FITARA.db")
   
   #CREATING A ROW OBJECT
   con.row_factory = sql.Row
   
   #CREATING A CURSOR
   cur = con.cursor()
   cur.execute("select * from fitt")
   
   #FETCH ALL THE ROWS
   rows = cur.fetchall();	
   
  
   return render_template("list.html",rows = rows)
@app.route('/addML', methods = ['POST'])
def addML():
	print("Here")




@app.route('/add', methods = ['POST'])
def add():
	
	#A function for preprocessing 
	MAX_NB_WORDS = 50000
	# Max number of words in each complaint.
	MAX_SEQUENCE_LENGTH = 250
	# This is fixed.
	EMBEDDING_DIM = 100
	#Get the input from user
	import pandas as pd
	if request.method=='POST':
		comment=request.form['comment']
	
	print(request.form['algo'])
	text=pd.Series(comment)	
	#Remoce \n \r and \t
	text=text.str.replace('\n','')
	text=text.str.replace('\r','')
	text=text.str.replace('\t','')
	  
	 #This removes unwanted texts
	text = text.apply(lambda x: re.sub(r'[0-9]','',x))
	text = text.apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
	  
	#Converting all upper case to lower case
	text= text.apply(lambda s:s.lower() if type(s) == str else s)
	  
	 #Remove un necessary white space
	text=text.str.replace('  ',' ')

	 #print(dataset.head(10))
	#Remove stop words
	nlp=spacy.load("en_core_web_sm")
	text =text.apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
	#Tokenize and padd
	if request.form['algo']=='AI':
		new_model = tf.keras.models.load_model('mymodel.h5')
		tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
		tokenizer.fit_on_texts(text)
		word_index = tokenizer.word_index
	

		X = tokenizer.texts_to_sequences(text)                         
		X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

		#Make prediction using the pre-trained model
		my_prediction = new_model.predict(X)

		#Get the probability of it being FITARA AFFECTED
		my_prediction=my_prediction[0][1]

	else:
		from sklearn.externals import joblib 
		
		nb = joblib.load('NaiveB.pkl')
		cv=joblib.load('cv.pkl')
	
		tr = cv.transform(text)
		print(tr)
		from sklearn.naive_bayes import MultinomialNB
		my_prediction=nb.predict_proba(tr)
		print("Predicton is " +str(my_prediction))
		my_prediction=my_prediction[0][1]

	
	my_prediction=float(my_prediction)

	answer=""
	if my_prediction>=0.6:
		answer="Yes"
	else:
		answer="No"
	try:
			#ESTABLISH CONNECTION
			comment = request.form['comment']
			with sql.connect("FITARA.db") as con:
				cur=con.cursor()
				#INSERT VALUES
				cur.execute("INSERT INTO fitt(doc,prediction,ans) VALUES (?,?,?)",(comment,str(my_prediction),answer))
				con.commit()
				msg = " Prediction is " + str(my_prediction)
				flag=1 
	except:
		#EXCEPTION
		con.rollback()
		msg="Error in inserting"
		print(msg)
	finally:
		#PASS THE VALUES TO THE RESULT.HTML PAGE
		return render_template("result.html",msg = msg)
	con.close()


if __name__ == '__main__':
	app.run(debug=True)
