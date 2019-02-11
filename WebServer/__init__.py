import string 
from flask import * 
from Database import *
from flask_sessionstore import Session
import json

#from HoneyServer.HoneyWell import *
from HoneyServer.Summarizer import *

#from socket import *
from jsonrpclib import Server # Install jsonrpclib-pelix, NOT jsonrpclib!

app = Flask(__name__)
app.config.update(
    DATABASE = 'Ardulous'
)
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
#Session(app)

import jinja2
import os
JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'])

#s = socket(AF_INET, SOCK_STREAM)
#s.connect(('0.0.0.0', 8194)) # Our socket for Calling Search Worker

conn = Server('http://localhost:1006') # RPC Server

global db  
db = Database("mongodb://localhost:27017/")

# Set the secret key to some random bytes. Keep this really secret!
import os 
import random
app.secret_key = os.urandom(32)#bytes(str(hex(random.getrandbits(128))), 'ascii')

@app.errorhandler(404)
def page_not_found(e):
    return render_template("/404.html")

@app.route("/logout", methods=["GET", "POST"])
def logout():
    global db
    del db 
    db = Database("mongodb://localhost:27017/")
    session.pop('login', None)
    session.pop('feedpos', None)
    return redirect("/login_user")#render_template("/login_user.html")

@app.route("/", methods=["GET", "POST"])        # Home Page
@app.route("/home", methods=["GET", "POST"]) 
def home():
    global db 
    if "login" in session: 
        ss = session['login']
    else:
        return render_template('/anon_dashboard.html')


@app.route("/search", methods=["GET", "POST"])
def search():
    #if "login" in session:
    if request.method == "POST":
        try: 
            squery = request.form['search']
            global db
            tt = "At sufficiently low temperatures, free protons will bind to electrons. However, the character of such bound protons does not change, and they remain protons. A fast proton moving through matter will slow by interactions with electrons and nuclei, until it is captured by the electron cloud of an atom. The result is a protonated atom, which is a chemical compound of hydrogen. In vacuum, when free electrons are present, a sufficiently slow proton may pick up a single free electron, becoming a neutral hydrogen atom, which is chemically a free radical. Such free hydrogen atoms tend to react chemically with many other types of atoms at sufficiently low energies. When free hydrogen atoms react with each other, they form neutral hydrogen molecules (H2), which are the most common molecular component of molecular clouds in interstellar space. "
            #s.send(squery)
            rres = conn.resolveQuery(squery)
            print("Query Sent!")
            #for i in range(0, 5):
            #    rres.append(gh[i])
            #for i in rres:
            #    i['result-text'] = Summarizer().summary(i['result-text'])
            #tg = Summarizer().summary(tt)
            jk = list()
            pk = list()
            for i in rres:
                if squery in i['result-text']:
                    jk.append(i)
                else:
                    pk.append(i)
            rres = list(jk + pk) # rank exact matches
            #print(rres)
            #print(jk)
            #print(pk)
            #rres = [{'result-text':tg, 'result-image':"asd", 'result-doc-link':'google.com', 'result-doc-name':'Testing', 'result-modified-date':'01-2-2019', 'result-id':"123"}]
            return render_template("/search.html", squery = json.dumps(squery), results = json.dumps({"data":rres, "type":"results"}))
        except Exception as e:
            return render_template("/500.html", error = e)
        return render_template("/search.html")
    return render_template("/search.html")
    #return login_user()
