# --*--coding:utf-8--*--
from wsgiref.simple_server import make_server
from subprocess import call
import os


def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    os.system('cd ~/file/tmp_gitlab_test/test ； git pull origin master')  # 切换到项目的目录，并且执行pull操作
    print('git pull finish')
    return ['welcome']


httpd = make_server('', 8009, application)  # 监听8009端口
print('Serving HTTP on port 8009...')
httpd.serve_forever()