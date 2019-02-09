from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer

def resolveQuery(query):
    # Add definition here 
    text = query + "IT WORKS LOLOLOLOL!" # Replace with suitable query -> Value
    data = list([{'result-text':text, 'result-image':"asd", 'result-doc-link':'google.com', 'result-doc-name':'Testing', 'result-modified-date':'01-2-2019', 'result-id':"123"}, {}])
    return data

server = SimpleJSONRPCServer(('localhost', 1006))
server.register_function(resolveQuery)
print("Start server")
server.serve_forever()