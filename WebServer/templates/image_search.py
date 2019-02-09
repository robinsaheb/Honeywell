<html>

<head>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link href="{{ url_for('static', filename='css/search.css') }}" rel="stylesheet">
</head>

<body>
    <div id="layout">
        <!-- Place General things like login/logout etc here-->
    </div>
    <div id="search-bar">
        <form class="example" action="/search" method="post" style="max-width:35vw;text-align:center">
            <input type="text" placeholder="Search.." value="{{ squery | safe }}" name="search">
            <button type="submit"><i class="fa fa-search"></i></button>
        </form>
    </div>
    <div id="results-panel">
        
    </div>
    <div id="right-bar">

    </div>
    <script>
        loadMiniResultCards('{{ results | safe }}', "id");
    </script> 
</body>

</html>