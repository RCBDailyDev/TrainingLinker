"""
FILE_NAME: main.py
AUTHOR: RCB
CREATED: 2024/3/30-15:08
DESC: 
"""
# This is a sample Python script.
import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
import gradio as gr
import util.common_util as util
import ToolsUI.tab_mgr as tm
import ToolsUI.training_linker.api_page as api_page
import ToolsUI.training_linker.tools_page as tools_page

GradioTemplateResponseOriginal = None


def UI(**kwargs):
    print("Launch TrainLinker")
    css = ''
    head = ''

    css_path = "./CSS/"
    if os.path.exists(css_path):
        for r, d, files in os.walk(css_path):
            for file in files:
                if file.endswith(".css"):
                    s_path = os.path.join(os.path.abspath(r), file)
                    with open(s_path, 'r', encoding='utf8') as f:
                        css += f.read() + "\n"
                        print("loading CSS  {}".format(file))

    css_path = "./CSS_Overlay/"
    if os.path.exists(css_path):
        for r, d, files in os.walk(css_path):
            for file in files:
                if file.endswith(".css"):
                    s_path = os.path.join(os.path.abspath(r), file)
                    with open(s_path, 'r', encoding='utf8') as f:
                        css += f.read() + "\n"
                        print("loading CSS-Overlay  {}".format(file))

    js_path = "./javascript/"
    if os.path.exists(js_path):
        for r, d, files in os.walk(js_path):
            for f in files:
                if f.endswith(".js"):
                    s_path = os.path.join(os.path.abspath(r), f)
                    with open(s_path, 'r', encoding='utf8') as file:
                        head += "<script>{}</script>\n".format(file.read())
                    print("Loaded JS ", f)

    interface = gr.Blocks(css=css, head="", title="DataSetTools")

    ## init pathmgr
    path_mgr = util.get_path_mgr()
    path_mgr.SetTrainCFGPath(os.path.join(os.getcwd(), "TrainCfgOrig"))
    path_mgr.SetPromptLibPath(os.path.join(os.getcwd(), "prompt_lib"))
    with interface:
        with gr.Row(elem_classes='my_color_theme'):
            with gr.Tab("ToolsPage"):
                tools_page.ToolsPage()
            with gr.Tab("ApiPage"):
                api_page.ApiPage()


        tm.get_tab_mgr().implUILogic()

    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    global GradioTemplateResponseOriginal
    GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{head}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
    launch_kwargs["allowed_paths"] = ["."]
    interface.launch(**launch_kwargs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()
    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
