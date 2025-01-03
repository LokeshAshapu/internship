import os
import json
from PIL import Image
import csv
import pandas as pd
import numpy as np
import nltk
import ssl
import random
from datetime import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
nltk.download('stopwords')
# Open a file to read from it
#this is not possible in my system
#with open('res.json', 'r') as file:
#        file_content = file.read()

intents= [
        {"tag": "greeting",
        "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
        "responses": ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"]
        },
        {"tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye"],
        "responses": ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
        },
        {"tag": "thanks",
        "patterns": ["Thanks", "Thank you", "That's helpful"],
        "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
        {"tag": "hours",
        "patterns": ["What hours are you open?", "What are your hours?", "When are you open?" ],
        "responses": ["We're open every day 9am-9pm", "Our hours are 9am-9pm every day"]
        },
        {"tag": "location",
        "patterns": ["What is your location?", "Where are you located?", "What is your address?", "Where is your restaurant situated?" ],
        "responses": ["We are on the intersection of London Alley and Bridge Avenue.", "We are situated at the intersection of London Alley and Bridge Avenue", "Our Address is: 1000 Bridge Avenue, London EC3N 4AJ, UK"]
        },
        {"tag": "payments",
        "patterns": ["Do you take credit cards?", "Do you accept Mastercard?", "Are you cash only?" ],
        "responses": ["We accept VISA, Mastercard and AMEX", "We accept most major credit cards"]
        },
        {"tag": "todaysmenu",
        "patterns": ["What is your menu for today?", "What are you serving today?", "What is today's special?"],
        "responses": ["Today's special is Chicken Tikka", "Our speciality for today is Chicken Tikka"]
        },
        {"tag": "deliveryoption",
        "patterns": ["Do you provide home delivery?", "Do you deliver the food?", "What are the home delivery options?" ],
        "responses": ["Yes, we provide home delivery through UBER Eats and Zomato?", "We have home delivery options through UBER Eats and Zomato"]
        },
        {"tag": "greeting",
        "patterns": ["Hi", "Hey", "Hello", "Good morning!", "Hey! Good morning", "Hey there", "Hey Janet", "Very good morning", "A very good morning to you", "Greeting", "Greetings to you"],
        "responses": ["Hello I'm Restrobot! How can I help you?", "Hi! I'm Restrobot. How may I assist you today?"]
        },
        {"tag": "book_table",
        "patterns": ["Book a table","Can I book a table?", "I want to book a table", "Book seat", "I want to book a seat", "Can I book a seat?", "Could you help me book a table", "Can I reserve a seat?", "I need a reservation", "Can you help me with a reservation", "Can I book a reservation", "Can i have a table?", "Help me reserve a table", "book table"],
        "responses": ["Yes sure I can help you with that! How many people are you booking for?", "Sure! How many people are you booking for?", "Yes, I can help you with that! How many people are you booking for?"]
        },
        {"tag": "available_tables",
        "patterns": ["How many seats are available?", "Available seats", "How many tables are available?", "Available tables", "Are there any tables available?", "What is the capacity of the restaurant", "Are there any available tables?", "Are there any seats left?", "I wanted to know if there are any tables available now", "May I know if you have any tables which I can book?"],
        "responses": ["We have several tables available. How many people are you booking for?", "There are plenty of tables available. How many seats do you need?", "Yes, we have tables available. How many people will be joining you?"]
        },
        {"tag": "goodbye",
        "patterns": ["cya", "I will leave now","See you later", "Goodbye", "Leaving now, Bye" , "Good bye dear", "Bye dear","I am Leaving", "Have a Good day", "cya later", "I gotta go now", "I gotta rush now", "Thank you, bye", "Bye", "Ok Bye", "Okay goodnight", "Have a good day ahead", "Have a great day", "Tata", "Take care"],
        "responses": ["It's been my pleasure serving you!", "Hope to see you again soon! Goodbye!", "Bye! Hope to see you again!"]
        },
        {"tag": "identity",
        "patterns": ["what is your name", "what should I call you", "whats your name?", "who are you", "Are you human?", "Am i talking to a bot", "Are you a bot", "Can i have your name please", "name"],
        "responses": ["You can call me Restrobot.", "I'm Restrobot!", "I'm Restrobot."]
        },
        {"tag": "hours",
        "patterns": ["when are you guys open", "what are your hours", "hours of operation", "hours", "what is the timing", "when are you open", "Are you open on all days?", "are you open now", "are you open on holidays", "are you guys open on all weekdays?", "working hours", "hours", "what are your working hours?"],
        "responses": ["We are open 10am-12am Monday-Friday!"]
        },
        {"tag": "menu",
        "patterns": ["Id like to order something", "whats on the menu", "could i get something to eat", "Im damn hungry", "I am hungry" ,"Show me the menu", "What food do you have", "wHat food are you offering?", "whats on the menu today?", "Let me see the menu", "menu"],
        "responses": ["Here is our menu: \n1. Margherita Pizza \n2. Caesar Salad \n3. Spaghetti Carbonara \n4. Grilled Chicken \n5. Chocolate Lava Cake"]
        },
        {"tag": "contact",
        "patterns": ["contact information", "how do we contact you", "how can i contact you", "can i get the contact details", "I wanna give some feedback", "how can i give some feedback?", "CAn you give me the contact of an executive?", "What is the help desk phone number?","Can you give me your number", "Can i get the customer care number?", "CAn I get help desk number"],
        "responses": ["You can contact us at contact@restaurantloki.com, our help desk number is +91 XXXXXXXXXX"]
        },
        {"tag": "address",
        "patterns": ["what is the location?","whats the location", "where are you located?", "where is the restaurant located?", "address", "whats the address?", "what is the address of the restaurant", "I am not able to locate you", "I cant find your location", "CAn i have the address of the restaurant", "how to reach there?", "Address?", "what is the address of this restaurant?"],
        "responses": ["You can locate us at Aindri's Restro, Phase 1, Rd Number 6, Whitefield, Bengaluru, Karnataka 560066"]
        },
        {"tag": "positive_feedback",
        "patterns": ["the noodles were amazing","i loved the food", "you did a good job", "Love the food", "Really love it", "Love the staff behavior", "my son devoured the brownie!", "pizza was so cheesy!", "perfectly baked", "we are very satisfied with the service", "the soup was a real game changer", "such delicious flavours", "so glad we discovered this place", "Me and my family is very satisfied with this service and food", "this place is awesome"],
        "responses": ["Thank you for your positive feedback!", "We are glad you enjoyed your experience!", "Thank you! We hope to see you again soon!"]
        },
        {"tag": "negative_feedback",
        "patterns": ["what the fuck is wrong with these noodles?", "The choco lava was so undercooked", "Ew such a waste of money man", "Prices are too high honestly", "I hate the menu, such less options", "too salty", "we were served cold food", "so disappointed", "the food is pathetic", "hate it", "my wife hates the food","eww", "hate the staff behavior", "i hate it", "hate the service","please train the staff properly", "This was such a waste of money", "Hate the staff", "Extremely dissatisfied", "disappointed", "so bad", "very bad", "disgusting food"],
        "responses": ["We're sorry to hear that you had a bad experience. We'll work on improving our service.", "Thank you for your feedback. We'll strive to do better.", "We apologize for the inconvenience. Your feedback is valuable to us."]
        },
        {"tag": "sanitation",
        "patterns": ["is it really safe to eat here?","Could you tell your COVID safety protocols?", "I would like to know about the cleanliness of the restaurant", "Please share your sanitization process", "I am concerned about the COVID related sanitization", "Is it safe to eat out in this pandemic?", "are you clean", "i am concerned about the cleanliness"],
        "responses": ["I understand your concern. Here are the WHO recommended COVID protocols we follow to ensure your safety: \n 1. All our staff are double masked 24x7. \n 2. All our staff is checked for fevers and other symptoms daily. \n 3. All surfaces are frequently sanitized. \n 4. We use this friendly bot to reduce physical closeness to the least!"]
        },
        {"tag": "offers",
        "patterns": ["Could you tell me the pocket friendly options?","Are there any discounts going on?", "Are there any special offers today?", "What about the festive offers?", "Could you please tell me which foods are on discount?", "are there any discounts", "are there any discount offers", "do you have any offers?", "what are the offers going on?", "what are the discounts available?"],
        "responses": ["Yes, we have some great offers today! You can get a 20% discount on all orders above $50.", "We have a special offer: Buy one get one free on all desserts!", "Currently, we are offering a 15% discount on all vegan dishes."]
        },
        {"tag": "vegan_enquiry",
        "patterns": ["Can I see the vegan option?","Do you have any vegan options??", "What is vegan in your menu?", "I am vegan.", "Do you also have vegan food?", "vegan", "is this restaurant vegan", "is this place vegan"],
        "responses": ["Sure! Here are our vegan options: \n1. Vegan Margherita Pizza \n2. Vegan Caesar Salad \n3. Vegan Spaghetti Carbonara \n4. Grilled Tofu \n5. Vegan Chocolate Cake"]
        },
        {"tag": "veg_enquiry",
        "patterns": ["Can I see the vegetarian options?","Do you have any vegetarian options??", "Please show me your best vegetarian foods", "I dont want to eat non veg", "I am vegetarian", "vegetarian", "is this place vegetarian?"],
        "responses": ["Sure! Here are our vegetarian options: \n1. Margherita Pizza \n2. Caesar Salad \n3. Spaghetti Carbonara \n4. Grilled Paneer \n5. Chocolate Lava Cake"]
        },
        {"tag": "recipe_enquiry",
        "patterns": ["Could you tell me more about recipe of this dish?", "What is the recipe of this dish?", "what are the ingredients of this dish?", "tell me the recipe"],
        "responses": ["Sure! Here is the recipe for this dish: \n1. Gather all the ingredients. \n2. Follow the cooking instructions step by step. \n3. Serve and enjoy your meal!", "I can help with that! Here are the ingredients and steps to prepare this dish: \n1. Ingredients: [list of ingredients]. \n2. Steps: [detailed steps].", "Certainly! Here is how you can make this dish: \n1. Prepare the ingredients. \n2. Follow these steps: [detailed steps]. \n3. Enjoy your homemade dish!"]
        },
        {"tag": "suggest",
        "patterns": ["what do you recommend?","do you have any suggestions","please suggest something","why don't you recommend me a dish", "help me choose what to order", "Surprise me!", "Do you have any special recommendations for me?", "What do you suggest?", "what is your suggestion", "what is a must try", "what should i try", "what should i eat?", "tell me what to order", "tell me what i should buy", "recommend me a dish", "suggest me a dish"],
        "responses": ["I recommend trying our Margherita Pizza, it's a customer favorite!", "How about our Grilled Chicken? It's delicious!", "You should try our Spaghetti Carbonara, it's highly recommended!", "Our Caesar Salad is a great choice if you're looking for something light and fresh.", "If you have a sweet tooth, our Chocolate Lava Cake is a must-try!"]
        },
        {"tag": "general",
        "patterns": ["okay","sure","cool","hmm", "fine", "thanks", "uhuh"],
        "responses": [":)", "Glad to serve you!", "Happy to help!", "Always happy to assist you!"]
        }
        ]
# Prepare data for training
patterns = []
tags = []
for intent in intents:
        for pattern in intent["patterns"]:
                patterns.append(pattern)
                tags.append(intent['tag'])

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, tags)

image = Image.open('res.png')

# Resize the image
resized_image = image.resize((300, 200))
# Display an image
st.image(resized_image)
st.title("Restaurant Chatbot")
st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation")
global response
global user_input
global timestamp
counter = 0
user_input = st.text_input("You: ")
if user_input:
        # Convert input to vector and predict tag
        input_vector = vectorizer.transform([user_input])
        predicted_tag = model.predict(input_vector)[0]
        # Get response from the intents based on the predicted tag
        for intent in intents:
                if intent['tag'] == predicted_tag:
                        response = random.choice(intent['responses'])
        #st.write(f"Bot: {response}")
        st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
#file.close()
counter = 0
def main():
        
        global counter
# Create a sidebar menu with options
        menu= ["Home", "Conversation History", "About","Intents used"]
        choice = st.sidebar.selectbox ("Menu", menu)
# Home Menu
        if choice == "Home":
# Check if the chat_log.csv file exists, and if not, create it with column names
                if not os.path.exists('chat_log.csv'):
                        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
                                counter += 1
                                timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
                                # Save the user input and chatbot response to the chat_log.csv file
                        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow([user_input, 'response', timestamp])
                        if response.lower() in ['goodbye', 'bye']:
                                st.write("Thank you for chatting with me. Have a great day!")
                                st.stop()
        elif choice == "Conversation History":
                timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
                # Read the conversation history from the CSV file
                st.header("Conversation History")
                with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                        csv_reader = csv.reader (csvfile)
                        print(csv_reader)
#Skip the header row for row in csv_reader:
                        st.text(f"User: {user_input}")
                        if 'response' in globals():
                                st.text(f"Chatbot: {response}")
                        else:
                                st.text("Chatbot: No response available")
                        st.text(f"Timestamp: {timestamp}")
                        st.markdown ("-")
        elif choice == "About":
                st.write("The goal of this project is to create a chatbot that can understand and respond to a conversation")
                st.subheader("Problem Statement:")
                st.write("""The aim is to craft an advanced Restaurants Chatbot designed for a 5-star guest experience. This virtual assistant will handle orders,
table reservations, special offers, discounts and general inquiries. Using Natural Language Processing (NLP), it will understand guest questions and provide accurate,
conversational responses. The chatbot ensures seamless interaction, enhancing convenience and luxury. It’s a step toward redefining hospitality with intelligent automation.""")
                st.subheader("Problem Solution:")
                st.write("""Restaurants face the challenge of managing guest requests 24/7 without overwhelming staff. A chatbot streamlines this by automating responses to common inquiries,
improving efficiency. This enhances the guest experience while lightening the operational load, and which can leads the restaurant growth.""")
                st.subheader ("Project Overview:")
                st.write("""
the key requirements for implementing a chatbot using NLP in 3 lines:
1.	Strong NLP foundation: Accurate intent recognition, entity extraction, and fluent NLG.
2.	High-quality data: Ample, diverse, and well-annotated data for training.
3.	User-centric design: Prioritize user experience, accessibility, and continuous improvement.
""")

                st.subheader("Dataset:")
                st.write("""
The dataset used in this project is a collection of labelled intents and entities. The - Intents: The intent of the user input (e.g. "greeting", "booking slots", "about")
- Entities: The entities extracted from user input (e.g. "Hi", "is it really safe to eat here?","Could you tell your COVID safety protocols?"- Text: The user input text.
""")
                st.subheader("Streamlit Chatbot Interface:")
                st.write("The chatbot interface is built using Streamlit.")
                st.subheader("Conclusion: ")
                st.write("In this project, a chatbot is built that can understand and respond to users")
        
        elif choice=="Intents used":
                st.subheader("Intents are:")
                df = pd.DataFrame(intents)
                st.write(df[['tag', 'patterns']])

if __name__=='__main__':
        main()
