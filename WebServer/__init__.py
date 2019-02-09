import string 
from flask import * 
from Database import *
from flask_sessionstore import Session
import json

app = Flask(__name__)
app.config.update(
    DATABASE = 'Ardulous'
)
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
#Session(app)

global db  
db = Database("mongodb://localhost:27017/")

# Set the secret key to some random bytes. Keep this really secret!
import os 
import random
app.secret_key = os.urandom(32)#bytes(str(hex(random.getrandbits(128))), 'ascii')

@app.errorhandler(404)
def page_not_found(e):
    return render_template("/404.html")

@app.route("/", methods=["GET", "POST"])        # Home Page
@app.route("/home", methods=["GET", "POST"]) 
def home():
    global db 
    if "login" in session: 
        ss = session['login']
    else:
        return render_template('/anon_dashboard.html')
