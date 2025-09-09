from PIL import Image
import google.generativeai as genai
#give the api key
genai.configure(api_key="AIzaSyCY7sPtFcRomLj2JALGHkg3OYOr4gog5Cc")
#specify the model
model = genai.GenerativeModel("gemini-1.5-flash")

img = Image.open("mypic.jpg")
response=model.generate_content(
    [img,"write a cretive and short caption for this image"]
)
print(response)
