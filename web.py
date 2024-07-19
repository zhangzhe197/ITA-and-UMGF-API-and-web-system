import flask,requests,pdb,re,time,random,io
from PIL import Image
from flask import Flask, redirect,request, render_template
app = Flask(__name__)
def ITAapi(sentence, file):
    url = "http://172.24.239.44:8000/predict"
    data = {'sentence' : sentence}
    res = requests.post(url, data=data, files=file)
    return res
def UMGFapi(sentence, file):
    url = "http://172.24.239.44:8001/predict"
    data = {'sentence' : sentence}
    file = {'image' : file}
    res = requests.post(url, data=data,files=file)
    return res
def return_error_message(mess):
    return render_template("main.html", res= False , msg = mess)
def is_english(text):
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return False
    return True
def proess_with_the_tag(tag):
    return tag.split("-")[-1]
def image2byte(image):
    '''
    图片转byte
    image: 必须是PIL格式
    image_bytes: 二进制
    '''
    # 创建一个字节流管道
    img_bytes = io.BytesIO()
    #把PNG格式转换成的四通道转成RGB的三通道，然后再保存成jpg格式
    image = image.convert("RGB")
    # 将图片数据存入字节流管道， format可以按照具体文件的格式填写
    image.save(img_bytes, format="JPEG")
    # 从字节流管道中获取二进制
    image_bytes = img_bytes.getvalue()
    return image_bytes
@app.route('/main', methods=['POST', 'GET'])
def display():
    if request.method != 'POST':
        return render_template("main.html", res = False)
    else:
        usingExample =  request.form.get("example")
        # if usingExample == '1':
        #     model_chosen = request.form.get('model')
        #     time.sleep(0.2 + random.random() / 5)
        #     if model_chosen == "ITA":
        #         res = {"Tokens": ["Donald", "Trump", "was", "shot", "and", "wounded", "at", "a", "Republican", "campaign", "event", "in", "Pennsylvania", "."],
        #                 "Entity_type": ["B-PER", "E-PER", "O", "O", "O", "O", "O", "O", "S-ORG", "O", "O", "O", "S-LOC", "O"],
        #                 "Score": ["0.9990789890289307", "0.9999717473983765", "0.9999885559082031", "0.99685138463974", "0.9965479969978333", "0.995205283164978", "0.9981427192687988", "0.9929554462432861", "0.9172472953796387", "0.9967145919799805", "0.9832149147987366", "0.9958447813987732", "0.9992166757583618", "0.9999798536300659"]}
        #         return render_template('main.html', words = res['Tokens'] ,
        #                                 entity = res['Entity_type'],
        #                                 score = res['Score'],
        #                                 res = True)
        #     elif model_chosen == 'UMGF':
        #         res = {"Entity_type":["B-PER","I-PER","O","O","O","O","O","O","B-ORG","O","O","O","B-LOC"],
        #             "Tokens":["Donald","Trump","was","shot","and","wounded","at","a","Republican","campaign","event","in","Pennsylvania"]}
        #         return render_template('main.html', words = res['Tokens'] , 
        #                                 entity = res['Entity_type'],
        #                                 score = ["N\A"] * len(res['Tokens']),
        #                                 res = True)

        sentence = request.form.get("text")
        print("+=====================================+")
        print(f"|sentence to ask {sentence}")
        print("+=====================================+")
        if sentence == '': 
            return return_error_message("ERROR , you must input one sentence")
        elif not is_english(sentence):
            return return_error_message('ERROR , Only supports English entity recognition')
        else: 
            pass
        
        file_name = request.files['image'].filename 
        if file_name == '' or file_name[-3:] in ['jpg', 'png']:
            pass
        else:
            return return_error_message("ERROR , you must upload a image file")
        model_chosen = request.form.get('model')
        if model_chosen == "ITA":
            res = ITAapi(sentence, request.files['image']).json()
            return render_template('main.html', words = res['Tokens'] ,
                                    entity = res['Entity_type'],
                                      score = res['Score'],
                                      res = True)
        elif model_chosen == 'UMGF' : 
            if request.files['image'].filename == '' and usingExample == '0':
                return return_error_message("ERROR , you must give a picture when using UMGF")
            elif usingExample == '1':
                res = UMGFapi(sentence, image2byte(Image.open("./static/trump.jpg"))).json()
            else : 
                res = UMGFapi(sentence,request.files['image']).json()
            return render_template('main.html', words = res['Tokens'] , 
                                      entity = res['Entity_type'],
                                     score = ["N\A"] * len(res['Tokens']),
                                     res = True)
        else :
            return return_error_message("ERROR , you must choose one model")
    
        
if __name__ == '__main__':
    app.run(port=5000)
