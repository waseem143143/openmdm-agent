"""
Generate 20 records (10 pairs) engineered to score exactly 0.95-0.98
so they queue for the AI Steward Agent review.

Each pair uses a real-world ambiguity pattern:
 - Single-letter typo in last name (same Soundex = same blocking bucket)
 - Address abbreviation difference
 - Slight formatting variation in phone

Fixes applied from score verification:
 - Phillips/Philips:  SSN added to PHARMACY (was 0.86, now 0.96)
 - Anderson/Andersen: SSN added to PHARMACY (was 0.85, now 0.95)
 - Bennett/Benet:     Email added to CLAIMS  (was 0.95, now 0.97)
"""

import csv, os

FIELDNAMES = [
    "record_id","source_system","source_priority","first_name","last_name",
    "middle_name","date_of_birth","gender","ssn_last4","address_line1",
    "city","state","zip_code","phone","email","mrn","insurance_id",
    "record_created","record_updated"
]

ambiguous_records = [

    # ── Pair 1: Williams / Wiliams (double-l drop) ── expected score ~0.959
    {"record_id":"AMB001","source_system":"EHR","source_priority":1,
     "first_name":"Robert","last_name":"Williams","middle_name":"James",
     "date_of_birth":"1971-06-15","gender":"M","ssn_last4":"3341",
     "address_line1":"2210 Birchwood Avenue","city":"Columbus","state":"OH",
     "zip_code":"43201","phone":"614-772-0193","email":"rwilliams@gmail.com",
     "mrn":"MRN-33401","insurance_id":"",
     "record_created":"2018-03-10","record_updated":"2024-01-15"},

    {"record_id":"AMB002","source_system":"CLAIMS","source_priority":3,
     "first_name":"Robert","last_name":"Wiliams","middle_name":"",
     "date_of_birth":"1971-06-15","gender":"M","ssn_last4":"3341",
     "address_line1":"2210 Birchwood Ave","city":"Columbus","state":"OH",
     "zip_code":"43201","phone":"6147720193","email":"",
     "mrn":"MRN-33401","insurance_id":"INS-OH-5512",
     "record_created":"2019-07-22","record_updated":"2024-02-28"},

    # ── Pair 2: Mitchell / Mitchel (double-l drop) ── expected score ~0.960
    {"record_id":"AMB003","source_system":"EHR","source_priority":1,
     "first_name":"Sandra","last_name":"Mitchell","middle_name":"Lynn",
     "date_of_birth":"1983-11-02","gender":"F","ssn_last4":"7892",
     "address_line1":"88 Lakeview Drive","city":"Memphis","state":"TN",
     "zip_code":"38101","phone":"901-345-0781","email":"smitchell@yahoo.com",
     "mrn":"MRN-44512","insurance_id":"",
     "record_created":"2020-01-08","record_updated":"2023-11-30"},

    {"record_id":"AMB004","source_system":"CLAIMS","source_priority":3,
     "first_name":"Sandra","last_name":"Mitchel","middle_name":"",
     "date_of_birth":"1983-11-02","gender":"F","ssn_last4":"7892",
     "address_line1":"88 Lakeview Dr","city":"Memphis","state":"TN",
     "zip_code":"38101","phone":"9013450781","email":"",
     "mrn":"MRN-44512","insurance_id":"INS-TN-3301",
     "record_created":"2021-03-14","record_updated":"2024-03-01"},

    # ── Pair 3: Phillips / Philips (double-l drop) ── SSN added → score ~0.960
    {"record_id":"AMB005","source_system":"EHR","source_priority":1,
     "first_name":"Kevin","last_name":"Phillips","middle_name":"Scott",
     "date_of_birth":"1965-09-28","gender":"M","ssn_last4":"4421",
     "address_line1":"501 Riverside Road","city":"Louisville","state":"KY",
     "zip_code":"40201","phone":"502-891-0334","email":"kphillips@work.com",
     "mrn":"MRN-55234","insurance_id":"",
     "record_created":"2016-05-19","record_updated":"2023-12-20"},

    {"record_id":"AMB006","source_system":"PHARMACY","source_priority":5,
     "first_name":"Kevin","last_name":"Philips","middle_name":"",
     "date_of_birth":"1965-09-28","gender":"M","ssn_last4":"4421",
     "address_line1":"501 Riverside Rd","city":"Louisville","state":"KY",
     "zip_code":"40201","phone":"502-891-0334","email":"",
     "mrn":"","insurance_id":"INS-KY-8821",
     "record_created":"2022-08-07","record_updated":"2024-01-10"},

    # ── Pair 4: Harrison / Harison (double-r drop) ── expected score ~0.959
    {"record_id":"AMB007","source_system":"EHR","source_priority":1,
     "first_name":"Patricia","last_name":"Harrison","middle_name":"Ann",
     "date_of_birth":"1958-03-17","gender":"F","ssn_last4":"6634",
     "address_line1":"1475 Magnolia Boulevard","city":"Shreveport","state":"LA",
     "zip_code":"71101","phone":"318-220-0567","email":"pharrison@email.com",
     "mrn":"MRN-66781","insurance_id":"",
     "record_created":"2014-09-03","record_updated":"2023-10-15"},

    {"record_id":"AMB008","source_system":"CLAIMS","source_priority":3,
     "first_name":"Patricia","last_name":"Harison","middle_name":"",
     "date_of_birth":"1958-03-17","gender":"F","ssn_last4":"6634",
     "address_line1":"1475 Magnolia Blvd","city":"Shreveport","state":"LA",
     "zip_code":"71101","phone":"3182200567","email":"",
     "mrn":"MRN-66781","insurance_id":"INS-LA-4490",
     "record_created":"2015-11-25","record_updated":"2024-02-14"},

    # ── Pair 5: Sullivan / Sulivan (double-l drop) ── expected score ~0.959
    {"record_id":"AMB009","source_system":"EHR","source_priority":1,
     "first_name":"Thomas","last_name":"Sullivan","middle_name":"Edward",
     "date_of_birth":"1979-07-04","gender":"M","ssn_last4":"1123",
     "address_line1":"334 Oakdale Street","city":"Providence","state":"RI",
     "zip_code":"02901","phone":"401-553-0892","email":"tsullivan@email.com",
     "mrn":"MRN-77345","insurance_id":"",
     "record_created":"2019-02-28","record_updated":"2024-01-22"},

    {"record_id":"AMB010","source_system":"CLAIMS","source_priority":3,
     "first_name":"Thomas","last_name":"Sulivan","middle_name":"",
     "date_of_birth":"1979-07-04","gender":"M","ssn_last4":"1123",
     "address_line1":"334 Oakdale St","city":"Providence","state":"RI",
     "zip_code":"02901","phone":"4015530892","email":"",
     "mrn":"MRN-77345","insurance_id":"INS-RI-2210",
     "record_created":"2020-06-15","record_updated":"2024-03-10"},

    # ── Pair 6: Hoffmann / Hoffman (extra n) ── expected score ~0.964
    {"record_id":"AMB011","source_system":"EHR","source_priority":1,
     "first_name":"Angela","last_name":"Hoffmann","middle_name":"Rose",
     "date_of_birth":"1986-12-11","gender":"F","ssn_last4":"9934",
     "address_line1":"789 Cedar Lane","city":"Madison","state":"WI",
     "zip_code":"53701","phone":"608-441-0278","email":"ahoffmann@gmail.com",
     "mrn":"MRN-88123","insurance_id":"",
     "record_created":"2021-04-17","record_updated":"2024-02-05"},

    {"record_id":"AMB012","source_system":"CLAIMS","source_priority":3,
     "first_name":"Angela","last_name":"Hoffman","middle_name":"",
     "date_of_birth":"1986-12-11","gender":"F","ssn_last4":"9934",
     "address_line1":"789 Cedar Lane","city":"Madison","state":"WI",
     "zip_code":"53701","phone":"6084410278","email":"",
     "mrn":"MRN-88123","insurance_id":"INS-WI-6612",
     "record_created":"2022-08-30","record_updated":"2024-03-15"},

    # ── Pair 7: Anderson / Andersen (son→sen) SSN added → score ~0.954
    {"record_id":"AMB013","source_system":"EHR","source_priority":1,
     "first_name":"Christine","last_name":"Anderson","middle_name":"Marie",
     "date_of_birth":"1992-04-23","gender":"F","ssn_last4":"2278",
     "address_line1":"42 Pinehurst Avenue","city":"Omaha","state":"NE",
     "zip_code":"68101","phone":"402-667-0445","email":"canderson@email.com",
     "mrn":"MRN-99012","insurance_id":"",
     "record_created":"2022-01-09","record_updated":"2024-01-30"},

    {"record_id":"AMB014","source_system":"PHARMACY","source_priority":5,
     "first_name":"Christine","last_name":"Andersen","middle_name":"",
     "date_of_birth":"1992-04-23","gender":"F","ssn_last4":"2278",
     "address_line1":"42 Pinehurst Ave","city":"Omaha","state":"NE",
     "zip_code":"68101","phone":"402-667-0445","email":"",
     "mrn":"","insurance_id":"INS-NE-3345",
     "record_created":"2023-03-22","record_updated":"2024-02-18"},

    # ── Pair 8: Freeman / Freeman address apt variant ── score ~0.965
    {"record_id":"AMB015","source_system":"EHR","source_priority":1,
     "first_name":"Marcus","last_name":"Freeman","middle_name":"Dale",
     "date_of_birth":"1974-08-30","gender":"M","ssn_last4":"5567",
     "address_line1":"1901 Broadway","city":"New York","state":"NY",
     "zip_code":"10001","phone":"212-334-0789","email":"mfreeman@nyc.com",
     "mrn":"MRN-10234","insurance_id":"",
     "record_created":"2017-06-14","record_updated":"2023-11-05"},

    {"record_id":"AMB016","source_system":"CLAIMS","source_priority":3,
     "first_name":"Marcus","last_name":"Freeman","middle_name":"",
     "date_of_birth":"1974-08-30","gender":"M","ssn_last4":"5567",
     "address_line1":"1901 Broadway Apt 7C","city":"New York","state":"NY",
     "zip_code":"10001","phone":"2123340789","email":"",
     "mrn":"MRN-10234","insurance_id":"INS-NY-7712",
     "record_created":"2018-09-01","record_updated":"2024-03-20"},

    # ── Pair 9: Campbell / Cambel (double-p drop) ── expected score ~0.952
    {"record_id":"AMB017","source_system":"EHR","source_priority":1,
     "first_name":"Denise","last_name":"Campbell","middle_name":"Joy",
     "date_of_birth":"1968-01-19","gender":"F","ssn_last4":"8801",
     "address_line1":"655 Willow Creek Road","city":"Tucson","state":"AZ",
     "zip_code":"85701","phone":"520-778-0234","email":"dcampbell@email.com",
     "mrn":"MRN-21345","insurance_id":"",
     "record_created":"2015-12-01","record_updated":"2023-09-18"},

    {"record_id":"AMB018","source_system":"CLAIMS","source_priority":3,
     "first_name":"Denise","last_name":"Cambel","middle_name":"",
     "date_of_birth":"1968-01-19","gender":"F","ssn_last4":"8801",
     "address_line1":"655 Willow Creek Rd","city":"Tucson","state":"AZ",
     "zip_code":"85701","phone":"5207780234","email":"",
     "mrn":"MRN-21345","insurance_id":"INS-AZ-9901",
     "record_created":"2016-07-14","record_updated":"2024-01-25"},

    # ── Pair 10: Bennett / Benet + email in CLAIMS → score ~0.978
    {"record_id":"AMB019","source_system":"EHR","source_priority":1,
     "first_name":"Victor","last_name":"Bennett","middle_name":"Paul",
     "date_of_birth":"1956-05-07","gender":"M","ssn_last4":"3312",
     "address_line1":"2280 Spruce Avenue","city":"Richmond","state":"VA",
     "zip_code":"23201","phone":"804-556-0891","email":"vbennett@retired.com",
     "mrn":"MRN-32456","insurance_id":"",
     "record_created":"2013-08-22","record_updated":"2023-08-30"},

    {"record_id":"AMB020","source_system":"CLAIMS","source_priority":3,
     "first_name":"Victor","last_name":"Benet","middle_name":"",
     "date_of_birth":"1956-05-07","gender":"M","ssn_last4":"3312",
     "address_line1":"2280 Spruce Ave","city":"Richmond","state":"VA",
     "zip_code":"23201","phone":"8045560891","email":"vbennett@retired.com",
     "mrn":"MRN-32456","insurance_id":"INS-VA-4423",
     "record_created":"2014-10-11","record_updated":"2024-02-08"},
]

output_path = "data/ambiguous_patients.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(ambiguous_records)

print(f"Written {len(ambiguous_records)} records to {output_path}")
