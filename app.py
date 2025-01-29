import smtplib
from bson import ObjectId
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import densenet121
from transformers import ViTModel, ViTConfig
import torch.nn as nn
from email.mime.text import MIMEText
app = Flask(__name__)
app.secret_key = "your_secret_key"  
ADMIN_EMAIL = "admin1@gmail.com"

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["diabetic_retinopathy"]
patients_collection = db["patients"]
users_collection = db["users"]  
emails_collection = db['emails'] 

# Define the CNN-ViT model
class CNN_ViT_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_ViT_Model, self).__init__()
        self.cnn = densenet121(pretrained=True)
        self.cnn.classifier = nn.Identity()
        vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes)
        self.vit = ViTModel(vit_config)
        self.fc = nn.Sequential(
            nn.Linear(1024 + vit_config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(pixel_values=x).pooler_output
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        return self.fc(combined_features)

# Load and initialize the model
num_classes = 5
model = CNN_ViT_Model(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing for image inputs
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def ensure_admin_exists():
    admin_email = "admin1@gmail.com"
    admin_name = "admin1"
    admin_password = "admin123" 
    
    # Hash the password
    hashed_password = generate_password_hash(admin_password, method='pbkdf2:sha256', salt_length=8)

    # Check if the admin exists
    if not users_collection.find_one({"email": admin_email}):
        users_collection.insert_one({
            "_id": ObjectId("678d97b6cd25283e003bb103"),  
            "name": admin_name,
            "email": admin_email,
            "password": hashed_password,
            "role": "admin"
        })
        print("Admin user added successfully.")
    else:
        print("Admin user already exists.")

ensure_admin_exists()

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user"] = user["name"]
            session["user_id"] = str(user["_id"])  
            session["role"] = "admin" if email == ADMIN_EMAIL else "doctor"  

            # Redirect based on role
            if session["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("homepage"))  

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html", error=None)




@app.route('/logout')
def logout():
    session.clear()  # مسح جميع بيانات الجلسة
    return redirect(url_for("homepage"))

# Patient management routes
@app.route('/')
def homepage():
    is_doctor_logged_in = 'role' in session and session['role'] == 'doctor'
    return render_template('index1.html', is_doctor_logged_in=is_doctor_logged_in)


@app.route('/add_patient_page')
def add_patient_page():
    if "user" in session:
        return render_template('index2.html')
    return redirect(url_for("login_page"))


@app.route('/add_patient', methods=['POST'])
def add_patient():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    name = request.form.get('name')
    age = request.form.get('age')
    file = request.files.get('file')

    # Validate file presence and extension
    if not file or file.filename == '':
        return jsonify({"error": "No file provided"}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400

    try:
        # Process the image and get the diagnosis
        image = Image.open(file).convert('RGB')  # Validate the file as an image
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        # Map predictions to class labels
        classes = {
            0: "Normal",
            1: "Mild Diabetic Retinopathy",
            2: "Moderate Diabetic Retinopathy",
            3: "Severe Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }
        diagnosis = classes[int(predicted.item())]

        # Retrieve the correct doctor_id from the logged-in doctor's record
        doctor = users_collection.find_one({"_id": ObjectId(session.get("user_id"))})
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        doctor_id = doctor.get("doctor_id")  # Use the sequential doctor_id

        # Add the patient's data to the database
        patient_data = {
            "name": name,
            "age": age,
            "diagnosis": diagnosis,
            "doctor_id": doctor_id  # Save the sequential doctor_id
        }
        patients_collection.insert_one(patient_data)

        return jsonify({
            "message": "Patient added successfully!",
            "name": name,
            "age": age,
            "diagnosis": diagnosis,
            "doctor_id": doctor_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/patients', methods=['GET'])
def list_patients():
    if 'role' not in session or session['role'] != 'doctor':
        return redirect(url_for('login_page'))

    # Get the logged-in doctor's sequential ID
    doctor = users_collection.find_one({"_id": ObjectId(session.get("user_id"))})
    if not doctor:
        return redirect(url_for('login_page'))

    doctor_id = doctor.get("doctor_id")  # Use the sequential doctor_id

    # Fetch patients assigned to the logged-in doctor
    patients = list(patients_collection.find({"doctor_id": doctor_id}, {"_id": 0}))

    # Fetch appointments assigned to the logged-in doctor
    appointments = list(emails_collection.find({"doctor_id": doctor_id}, {"_id": 0}))

    return render_template('patients.html', patients=patients, appointments=appointments)



@app.route('/delete_patient', methods=['POST'])
def delete_patient():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    name = request.form.get('name')  
    if not name:
        return jsonify({"error": "Patient name not provided"}), 400

    try:
        doctor_id = session.get("user_id")
        result = patients_collection.delete_one({"name": name, "doctor_id": doctor_id})
        if result.deleted_count > 0:
            return jsonify({"success": True, "message": f"Patient {name} deleted successfully!"})
        else:
            return jsonify({"error": "Patient not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rediagnosis', methods=['GET', 'POST'])
def rediagnosis():
    if 'user' not in session or session['role'] != 'doctor':
        return redirect(url_for('login_page'))

    if request.method == 'GET':
        name = request.args.get('name')
        age = request.args.get('age')

        if not name or not age:
            return redirect(url_for('patients'))

        return render_template('rediagnosis.html', name=name, age=age)

    elif request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        file = request.files.get('file')

        if not file or file.filename == '':
            return jsonify({"error": "No file provided"}), 400

        # Process the image, make a diagnosis, and update the database
        try:
            image = Image.open(file).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)

            # Map predictions to class labels
            classes = {
                0: "Normal",
                1: "Mild Diabetic Retinopathy",
                2: "Moderate Diabetic Retinopathy",
                3: "Severe Diabetic Retinopathy",
                4: "Proliferative Diabetic Retinopathy"
            }
            new_diagnosis = classes[int(predicted.item())]

            # Update the patient's diagnosis in the database
            patients_collection.update_one(
                {"name": name, "age": age},
                {"$set": {"diagnosis": new_diagnosis}}
            )

            return jsonify({
                "name": name,
                "age": age,
                "diagnosis": new_diagnosis
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500



@app.route('/admin', methods=['GET'])
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login_page'))

    # Fetch all users with the role of "doctor"
    doctors = list(users_collection.find({"role": "doctor"}, {"_id": 1, "doctor_id": 1, "name": 1, "email": 1}))

    
    for doctor in doctors:
        doctor["_id"] = str(doctor["_id"])

    return render_template('admin_dashboard.html', doctors=doctors)




#delete_doctor
@app.route('/admin/delete_doctor/<string:email>', methods=['POST'])
def delete_doctor(email):
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login_page'))

    # Find the doctor by email
    doctor = users_collection.find_one({"email": email})
    if not doctor:
        return redirect(url_for('admin_dashboard', message="Doctor not found!", is_error=True))

    doctor_id = doctor.get("doctor_id")  # Get the sequential doctor_id

    # Delete the doctor
    result = users_collection.delete_one({"email": email})
    if result.deleted_count > 0:
        # delete all patients associated with the doctor
        patients_deleted = patients_collection.delete_many({"doctor_id": doctor_id})
        print(f"Deleted {patients_deleted.deleted_count} patients associated with doctor {doctor_id}")

        # delete all appointments associated with the doctor
        appointments_deleted = emails_collection.delete_many({"doctor_id": doctor_id})
        print(f"Deleted {appointments_deleted.deleted_count} appointments associated with doctor {doctor_id}")

        return redirect(url_for('admin_dashboard', message="Doctor and associated records deleted successfully!", is_error=False))
    else:
        return redirect(url_for('admin_dashboard', message="Failed to delete the doctor!", is_error=True))


#delete_report
@app.route('/admin/delete_report/<string:patient_name>', methods=['POST'])
def delete_report(patient_name):
    if 'user' not in session or session.get('user') != 'admin1':
        return redirect(url_for('login_page'))

    # Find the doctor ID associated with the patient
    patient = patients_collection.find_one({"name": patient_name})
    if not patient:
        return redirect(url_for('admin_dashboard', message="Patient not found!", is_error=True))

    doctor_id = patient.get("doctor_id")
    if not doctor_id:
        return redirect(url_for('admin_dashboard', message="Doctor not associated with this patient!", is_error=True))

    # Delete the patient record
    result = patients_collection.delete_one({"name": patient_name})
    if result.deleted_count > 0:
        # Redirect back to the doctor_details page with the current doctor_id
        return redirect(url_for('doctor_details', doctor_id=doctor_id, message=f"Report for patient {patient_name} deleted successfully!", is_error=False))

    return redirect(url_for('doctor_details', doctor_id=doctor_id, message="Patient not found!", is_error=True))


@app.route('/admin/add_doctor', methods=['GET', 'POST'])
def add_doctor():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if the doctor already exists
        if users_collection.find_one({"email": email}):
            return render_template("add_doctor.html", error="Doctor already exists")

        # Generate a new doctor ID
        counter = db.counters.find_one_and_update(
            {"_id": "doctor_id"},  
            {"$inc": {"seq": 1}},  
            upsert=True, 
            return_document=True  
        )

       
        new_doctor_id = f"{counter['seq']:03d}"

        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

       
        users_collection.insert_one({
            "doctor_id": new_doctor_id,
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": "doctor"
        })

        return redirect(url_for('admin_dashboard'))

    return render_template("add_doctor.html", error=None)



# Route to edit a doctor
@app.route('/admin/edit_doctor/<string:email>', methods=['GET', 'POST'])
def edit_doctor(email):
    if 'user' not in session or session.get('user') != 'admin1':
        return redirect(url_for('login_page'))
    
    doctor = users_collection.find_one({"email": email})
    if not doctor:
        return "Doctor not found", 404

    if request.method == 'POST':
        name = request.form.get('name')
        new_email = request.form.get('email')

        users_collection.update_one(
            {"email": email},
            {"$set": {"name": name, "email": new_email}}
        )
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_doctor.html', doctor=doctor)

# Route to edit a patient report
@app.route('/admin/edit_report/<string:patient_name>', methods=['GET', 'POST'])
def edit_report(patient_name):
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login_page'))

    report = patients_collection.find_one({"name": patient_name})
    if not report:
        return "Report not found", 404

    # Fetch all doctors for the dropdown
    doctors = list(users_collection.find({"role": "doctor"}, {"_id": 1, "name": 1, "doctor_id": 1}))

    if request.method == 'POST':
        name = request.form.get('name')
        diagnosis = request.form.get('diagnosis')
        doctor_id = request.form.get('doctor_id')

        # Update the patient's report
        patients_collection.update_one(
            {"name": patient_name},
            {"$set": {"name": name, "diagnosis": diagnosis, "doctor_id": doctor_id}}
        )

        # Redirect back to the admin dashboard
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_report.html', report=report, doctors=doctors)

    
#delete_email
@app.route('/admin/delete_email/<string:email>', methods=['POST'])
def delete_email(email):
    if 'user' not in session or session.get('user') != 'admin1':
        return redirect(url_for('admin_dashboard', message="Unauthorized action!", is_error=True))

    # Fetch the doctor_id associated with the email
    appointment = emails_collection.find_one({"email": email})
    if not appointment:
        return redirect(url_for('admin_dashboard', message="Appointment not found!", is_error=True))

    doctor_id = appointment.get("doctor_id")
    if not doctor_id:
        return redirect(url_for('admin_dashboard', message="Doctor not associated with this appointment!", is_error=True))

    # Delete the email
    result = emails_collection.delete_one({"email": email})
    if result.deleted_count > 0:
        return redirect(url_for('doctor_details', doctor_id=doctor_id, message=f"Email {email} deleted successfully!", is_error=False))
    return redirect(url_for('doctor_details', doctor_id=doctor_id, message="Email not found!", is_error=True))


#reschedule_email
@app.route('/admin/reschedule_email/<string:email>', methods=['POST'])
def reschedule_email(email):
    if 'user' not in session or session.get('user') != 'admin1':
        return redirect(url_for('admin_dashboard', message="Unauthorized action!", is_error=True))

    new_date = request.form.get('new_date')
    if not new_date:
        return redirect(url_for('admin_dashboard', message="New date is required!", is_error=True))

    # Fetch the doctor_id associated with the appointment
    appointment = emails_collection.find_one({"email": email})
    if not appointment:
        return redirect(url_for('admin_dashboard', message="Appointment not found!", is_error=True))

    doctor_id = appointment.get("doctor_id")
    if not doctor_id:
        return redirect(url_for('admin_dashboard', message="Doctor not associated with this appointment!", is_error=True))

    # Update the appointment date
    result = emails_collection.update_one(
        {"email": email},
        {"$set": {"appointment_date": new_date}}
    )

    if result.matched_count > 0:
        return redirect(url_for('doctor_details', doctor_id=doctor_id, message=f"Appointment for {email} rescheduled successfully!", is_error=False))
    return redirect(url_for('doctor_details', doctor_id=doctor_id, message="Email not found!", is_error=True))


@app.route('/admin/doctor_details/<string:doctor_id>', methods=['GET'])
def doctor_details(doctor_id):
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login_page'))

    # Fetch the doctor's details using the sequential doctor_id
    doctor = users_collection.find_one(
        {"doctor_id": doctor_id},  
        {"_id": 1, "doctor_id": 1, "name": 1, "email": 1, "role": 1}
    )

    if not doctor:
        return redirect(url_for('admin_dashboard'))  # Redirect if doctor is not found

    # Fetch patients and appointments related to this doctor
    patients = list(
        patients_collection.find({"doctor_id": doctor_id}, {"_id": 0, "name": 1, "age": 1, "diagnosis": 1})
    )
    appointments = list(
        emails_collection.find({"doctor_id": doctor_id}, {"_id": 0, "name": 1, "email": 1, "appointment_date": 1})
    )

    return render_template('doctor_details.html', doctor=doctor, patients=patients, appointments=appointments)








    
#book oppointment
@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        # Get data from the form
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        appointment_date = request.form.get('date')

        if 'role' not in session or session['role'] != 'doctor':
            return jsonify({"error": "Unauthorized"}), 401

        # Get the logged-in doctor's ID
        doctor = users_collection.find_one({"_id": ObjectId(session.get("user_id"))})
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        # Retrieve the correct doctor_id 
        doctor_id = doctor.get("doctor_id")

        # Validate that the required fields are not empty
        if not name or not email:
            return jsonify({"success": False, "message": "Name and Email are required!"}), 400

        # Save the data to MongoDB
        email_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "appointment_date": appointment_date,
            "doctor_id": doctor_id  # Use the correct doctor_id
        }
        emails_collection.insert_one(email_data)

        return jsonify({"success": True, "message": "Your details have been recorded successfully."})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"success": False, "message": "An error occurred. Please try again later."}), 500


if __name__ == '__main__':
    app.run(debug=True)

