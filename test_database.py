from database.database import create_tables, register_student

create_tables()

success, message = register_student(
    "Charan",
    "charan@test.com",
    "123456",
    "Aditya University",
    "CSE",
    "2nd Year",
    "25B11CS625"
)

print(message)