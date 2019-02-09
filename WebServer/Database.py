import pymongo
from bson.objectid import ObjectId
import hashlib
import json
from werkzeug.security import generate_password_hash, check_password_hash

import re

class Database:
    def __init__(self, url):
        self.client = pymongo.MongoClient(url)
        self.db = self.client['default']
        return 

    def createUser(self, data, type = "normaluser"):
        d = self.db 
        try:
           # b = d['users'][data['id']]
            if(d['users'].find_one({'_id':data['id']})):
               return 2
            dd = {"_id":data['id'], "password" : generate_password_hash(data['password']), "type":type, "email":data['email'], "feed": [], "originals":[], "connections":{'friends':[], 'followers':[], 'following':[]}, "personal":{"profile_pic":data['profile_pic'], "profile_cover":data['profile_cover'], "name": data['name'], "info": data['info'], "dob":data['dob'], "city": data['city'], "address":data['address'], "occupation":data['occupation'], "interest":data['interest']} }
            d['user'].insert_one(dd)
        except:
            return False 
        return False

    def validateUser(self, uid, upass):
        d = self.db
        try: 
            h = d['users'].find_one({"_id":uid})['password']
            if(check_password_hash(h, upass)):
                return True
            return False
        except:
            return False

    def validateAdmin(self, uid, upass):
        d = self.db
        try: 
            h = d['users'].find_one({"_id":uid})['password']
            if d['users'].find_one({"_id":uid})['type'] == 'admin':
                if(check_password_hash(h, upass)):
                    return True
            return False
        except:
            return False
