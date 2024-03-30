from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from tensorflow import keras
from PIL import Image
import numpy as np
from pathlib import Path

app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'  # Update with your MySQL server hostname or IP address
app.config['MYSQL_USER'] = 'root'  # Update with your MySQL username
app.config['MYSQL_PASSWORD'] = ''  # Update with your MySQL password
app.config['MYSQL_DB'] = 'message_db'

mysql = MySQL()
mysql.init_app(app)

# Load the trained model
raw_image_model = keras.models.load_model('base_model.h5')

# Dictionary containing information about herbs
herb_info = {
    1: ["Amla (Amalaki)", "Amla, also known as Amalaki in Ayurveda, is rich in vitamin C, which supports the immune system and improves digestion. It enhances hair and skin health, acting as a natural rejuvenator. Amla's antioxidant properties help combat free radicals and promote overall well-being.", "Fruit"],
    2: ["Adulsa", "Adulsa, often referred to as Vasaka in Ayurveda, is highly effective for respiratory conditions, including cough, bronchitis, and asthma. It possesses bronchodilator and anti-inflammatory properties, making it a key remedy for respiratory health. Additionally, it aids in wound healing.", "Leaves"],
    3: ["Brahmi", "Brahmi is renowned for enhancing cognitive function and supporting brain health. It is known to reduce stress and anxiety levels, improving mental clarity and focus. As a memory-enhancing herb, Brahmi is a valuable ally for mental well-being.", "Leaves"],
    4: ["Dashmool", "Dashmool is a versatile combination of ten different roots, each contributing to its broad spectrum of healing properties. This Ayurvedic formulation is particularly valued for its potent anti-inflammatory effects and its ability to balance Vata and Kapha doshas.", "Roots"],
    5: ["Gokhshura", "Gokshura, also called Gokhru in Hindi, is known for its diuretic and aphrodisiac properties. It supports urinary tract health, making it beneficial for kidney and bladder function. Traditionally, it has been used to enhance vitality and energy levels.", "Fruit, Roots"],
    6: ["Guduchi (Giloy)", "Guduchi, commonly known as Giloy, is celebrated for its immune-boosting properties. It promotes general health, reduces fever during illnesses, and supports liver function. Guduchi also supports skin health, making it beneficial for conditions like eczema.", "Stems, Leaves"],
    7: ["Hirda", "Hirda is an Ayurvedic herb known for its digestive benefits, especially for alleviating digestive issues like indigestion and gas. It is also used for supporting heart health and overall well-being.", "Fruit"],
    8: ["Madanphal", "Madanphal is a versatile herb used for various medicinal purposes. It is especially known for its pain-relieving properties, making it a valuable natural remedy for managing conditions that cause pain and discomfort.", "Fruit"],
    9: ["Manjistha", "Highly regarded for blood purification, improving skin health, and reducing inflammation in Ayurveda. It is often used to treat skin disorders while enhancing complexion.", "Roots"],
    10: ["Nagarmotha", "Used as a digestive aid, Nagarmotha possesses anti-inflammatory properties and is effective in reducing stress and anxiety levels. It also aids in detoxification of the body.", "Roots"],
    11: ["Pippali", "Pippali is renowned for its digestive benefits, respiratory support, and metabolism-boosting properties. This warming herb stimulates digestion, enhances nutrient absorption, and supports overall vitality.", "Fruit"],
    12: ["Punarnava", "Punarnava acts as a diuretic and anti-inflammatory agent, making it a valuable herb for supporting kidney and liver health. It helps alleviate edema and promotes overall wellness.", "Roots"],
    13: ["Rasna", "Valued for its anti-inflammatory effects, Rasna is commonly used in arthritis treatment and pain relief. It possesses analgesic properties, providing relief from discomfort and pain.", "Leaves"],
    14: ["Shatavari", "Shatavari rejuvenates the body and helps balance hormones, particularly in women. It supports reproductive health and fertility and is a nourishing herb for overall well-being.", "Roots"],
    15: ["Vidarikand", "Known for rejuvenation and acting as a tonic, Vidarikand supports respiratory health, vitality, and physical strength. It's a cherished herb for enhancing endurance and overall well-being.", "Tubers"],
    16: ["Yashtimadhu (Licorice)", "Yashtimadhu possesses anti-inflammatory properties and provides soothing relief. It supports respiratory health and is used to soothe various ailments, including sore throat and cough.", "Roots"]
}

# Function to perform prediction on the uploaded image
def prediction(model, path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img)
    pred = np.argmax(predictions)
    return herb_info[pred][0], herb_info[pred][1], herb_info[pred][2]

@app.route('/submit_message', methods=['GET', 'POST'])
def submit_message():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Create MySQL cursor
        cursor = mysql.connection.cursor()

        # Execute the SQL query to insert the message into the database
        cursor.execute("INSERT INTO messages (name, email, message) VALUES (%s, %s, %s)", (name, email, message))

        # Commit changes to the database
        mysql.connection.commit()

        # Close the cursor
        cursor.close()

        return 'Message submitted successfully!'

    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return "No image uploaded"
    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected file"
    image_file.save('uploads/' + image_file.filename)
    image_path = 'uploads/' + image_file.filename
    name, description, useful_parts = prediction(raw_image_model, image_path)
    file_to_delete = Path(image_path)
    file_to_delete.unlink()
    return render_template('rawresult.html', name=name, desc=description, useful=useful_parts)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
