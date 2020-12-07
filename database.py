
import mysql.connector

mydb=mysql.connector.connect(
               host="us-cdbr-east-02.cleardb.com",
               user="b065bb4a6c47b3",
               passwd="592b2369",
               database="heroku_8bb5027e2cb7476"
               
	)

mycursor=mydb.cursor()

mycursor.execute("CREATE TABLE IF NOT EXISTS Owner_Data(id INT AUTO_INCREMENT PRIMARY KEY ,name VARCHAR(255),car_plate VARCHAR(255))")
 # mycursor.execute("CREATE TABLE IF NOT EXISTS Exhibito_Accounts(id  INT AUTO_INCREMENT PRIMARY KEY,email VARCHAR(255), password VARCHAR(255))")
