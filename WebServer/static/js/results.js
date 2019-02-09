
function createResultCard(data)
{
    var s = "";
    
    s = '<div class="container result-card"> <div class="result-area container"> <div class="result-text"> '+data['result-text']+' </div> <div class="result-image"> '+data['result-image']+' </div> </div> <div class="result-footer"> <div class="result-doc-name"> <a href="'+data['result-doc-link']+'">'+data['result-doc-name']+'</a> </div> <div class="result-modified-date"> '+data['result-doc-date']+' </div> <div class="result-feedback"> <div class="result-like" onclick="likeResult(\'' + data['result-id'] + '\')"></div> <div class="result-dislike" onclick="dislikeResult(\'' + data['result-id'] + '\')"></div> </div> </div> </div>'
    return s.replace(/undefined/g, "");
}

function createMultiResultCards(data)
{
    var dd = JSON.parse(data)['data'];
    var s = "";
    for(var i = 0; i < dd.length; i++)
    {
        s += createResultCard(dd[i]);
    }
    return s;
}

var feedpos = 0;

function loadResults(cc, offset) 
{
    var d = document.getElementById("result-panel");
    var fetchCount = cc;
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () 
    {
        if (this.readyState == 4 && this.status == 200) 
        {
            feedpos += fetchCount;
            d.innerHTML += createMultiResultCards(this.responseText);
        }
    };
    xhttp.open("POST", "/handlers/resultFetch", true);
    xhttp.send(JSON.stringify({count:fetchCount, feedpos:feedpos-offset}));
}

function loadMiniResultCards(obj, type)    // Called by Jinga2 at backend
{
    var dd = document.getElementById("results-panel");
    dd.innerHTML += createMultiResultCards(obj);
}   
