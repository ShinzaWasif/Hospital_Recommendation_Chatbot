import pandas as pd
import re

#import pandas as pd

hospitals = pd.read_csv("hospital.csv")   # aapka master hospital data
# Simulated DataFrame similar to what user showed

queries = pd.DataFrame([
    {"city": "Rawalpindi", "spec": "Multi-Disciplinary"},
    {"city": "Rawalpindi", "spec": "Medical"},
    {"city": "Rawalpindi", "spec": "Tourism"},
    {"city": "Karachi",    "spec": "Cardiology"},
    {"city": "Karachi",    "spec": "Dental"},
    {"city": "Karachi",    "spec": "Oncology"},
    {"city": "Karachi",    "spec": "Orthopedics"},
    {"city": "Karachi",    "spec": "Cosmetic Surgery"},
    {"city": "Lahore",     "spec": "Cancer Treatment"},
    {"city": "Peshawar",   "spec": "Cancer Treatment"},
    {"city": "Islamabad",  "spec": "Cardiology"},
    {"city": "Islamabad",  "spec": "Dental"},
    {"city": "Islamabad",  "spec": "Oncology"},
    {"city": "Islamabad",  "spec": "Orthopedics"},
    {"city": "Islamabad",  "spec": "Cosmetic Surgery"},
    {"city": "Islamabad",  "spec": "Pulmonology Transplant"},
    {"city": "Islamabad",  "spec": "General Surgery"},
    {"city": "Islamabad",  "spec": "Neurosurgery"},
    {"city": "Islamabad",  "spec": "Obstetrics And Gynaecology"},
    {"city": "Islamabad",  "spec": "Orthopedics Transplant"},
    {"city": "Peshawar",   "spec": "Cardiology"},
    {"city": "Peshawar",   "spec": "Dental"},
    {"city": "Peshawar",   "spec": "Orthopedics Transplant"},
    {"city": "Lahore",     "spec": "Cardiology"},
    {"city": "Lahore",     "spec": "Neurosurgery"},
    {"city": "Lahore",     "spec": "Obstetrics And Gynaecology"},
    {"city": "Lahore",     "spec": "Orthopedics"},
    {"city": "Lahore",     "spec": "Cosmetic Surgery"},
    {"city": "Rawalpindi", "spec": "Cardiology"},
    {"city": "Rawalpindi", "spec": "General Surgery"},
    {"city": "Rawalpindi", "spec": "Obstetrics And Gynaecology"},
    {"city": "Rawalpindi", "spec": "Urology"},
    {"city": "Rawalpindi", "spec": "Radiology"},
    {"city": "Lahore",     "spec": "General Surgery"},
    {"city": "Lahore",     "spec": "Pulmonology"},
    {"city": "Lahore",     "spec": "Gastroenterology"},
    {"city": "Lahore",     "spec": "Medicines"},
    {"city": "Islamabad",  "spec": "Pulmonology"},
    {"city": "Lahore",     "spec": "Cardiology Transplant"},
    {"city": "Lahore",     "spec": "Orthopedics"},
    {"city": "Lahore",     "spec": "Otolaryngology"},
    {"city": "Peshawar",   "spec": "Orthopedics"},
    {"city": "Peshawar",   "spec": "Pediatrics"},
    {"city": "Peshawar",   "spec": "Urology"},
    {"city": "Peshawar",   "spec": "Radiology"},
    {"city": "Peshawar",   "spec": "Pulmonology"},
    {"city": "Karachi",    "spec": "Orthopedics"},
    {"city": "Karachi",    "spec": "Obstetrics And Gynaecology"},
    {"city": "Karachi",    "spec": "General Surgery"},
    {"city": "Peshawar",   "spec": "Neurosurgery"},
    {"city": "Peshawar",   "spec": "ENT"},
    {"city": "Karachi",    "spec": "ENT"},
    {"city": "Karachi",    "spec": "Physiotherapy"},
    {"city": "Islamabad",  "spec": "Cosmetic Surgery"},
    {"city": "Peshawar",   "spec": "Cosmetic Surgery"},
    {"city": "Karachi",    "spec": "Reproductive Medicine"},
    {"city": "Karachi",    "spec": "Bone Marrow"},
    {"city": "Karachi",    "spec": "Transplant"},
    {"city": "Karachi",    "spec": "Hematology"},
    {"city": "Karachi",    "spec": "Cancer Treatment"},
    {"city": "Peshawar",   "spec": "Obesity"},
    {"city": "Peshawar",   "spec": "Ophthalmology"},
    {"city": "Peshawar",   "spec": "Dentistry"},
    {"city": "Lahore",     "spec": "Hip Replacement"},
    {"city": "Lahore",     "spec": "Knee Replacement"},
    {"city": "Karachi",    "spec": "Multi-Disciplinary"},
    {"city": "Karachi",    "spec": "Tertiary"},
    {"city": "Karachi",    "spec": "Care"},
    {"city": "Karachi",    "spec": "Renal Transplant"},
    {"city": "Karachi",    "spec": "Spinal Cord"},
    {"city": "Karachi",    "spec": "Orthopedics"}
])


records = []
for _, q in queries.iterrows():
    for _, h in hospitals.iterrows():
        has_spec = q["spec"].lower() in h["specialization"].lower()
        same_city = q["city"].lower() == h["city"].lower()
        label = int(has_spec and same_city)
        records.append({
            "query": f"{q['spec']} {q['city']}",
            "hospital_id": h["hospital_id"],
            "hospital_name": h["hospital_name"],
            "label": label
        })

df = pd.DataFrame(records)
df.to_csv("training_labels.csv", index=False)
