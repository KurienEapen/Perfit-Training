
<!DOCTYPE html>
<html>
<head>
 <title>Personal Details</title>
 <link rel="stylesheet" href="style.css">
 <style>
  table {
   border-collapse: collapse;
   width: 100%;
   color: #588c7e;
   font-family: monospace;
   font-size: 25px;
   text-align: left;
     } 
  th {
   background-color: #588c7e;
   color: white;
    }
  tr:nth-child(even) {background-color: #f2f2f2}
  .button {
  display: inline-block;
  border-radius: 4px;
  background-color: #f4511e;
  border: none;
  color: #FFFFFF;
  text-align: center;
  font-size: 20px;
  padding: 20px;
  width: 175px;
  transition: all 0.5s;
  cursor: pointer;
  margin-top: 5px;
  float : right;
}

.button span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button span:after {
  content: '\00bb';
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.button:hover span {
  padding-right: 25px;
}

.button:hover span:after {
  opacity: 1;
  right: 0;
  
  
}
.button1 {
  display: inline-block;
  border-radius: 4px;
  background-color: #f4511e;
  border: none;
  color: #FFFFFF;
  text-align: center;
  font-size: 20px;
  padding: 20px;
  width: 175px;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
}

.button1 span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.button1 span:after {
  content: '\00ab';
  position: absolute;
  opacity: 0;
  top: 0;
  left: -20px;
  transition: 0.5s;
}

.button1:hover span {
  padding-left: 25px;
}

.button1:hover span:after {
  opacity: 1;
  left: 0;
}
 </style>
</head>
<body>
 <table>
 <tr>
  <th>First Name</th> 
  <th>Last Name</th> 
  <th>Dob</th>
  <th>Blood Group</th>
  <th>Contact No</th>
 </tr>
 <?php
$conn = mysqli_connect("localhost", "id5401488_kurien", "123qwe", "id5401488_app_data");
  // Check connection
  if ($conn->connect_error) {
   die("Connection failed: " . $conn->connect_error);
  } 
 $username1 = trim($_REQUEST['username']);
 $password1 = trim($_REQUEST['password']);
  $sql = "SELECT * from PerfitLogin where username = '".$username1."' && password ='".$password1."'";
  $result = $conn->query($sql);
  if ($result->num_rows > 0) {
   // output data of each row
   while($row = $result->fetch_assoc()) {
    echo "<tr><td>" . $row["FirstName"]. "</td><td>" . $row["LastName"] . "</td><td>". $row["DoB"]. "</td><td>" . $row["BloodGroup"]. "</td><td>" . $row["ContactNo"]. "</td></tr>";
}
echo "</table>";
} else { echo "0 results"; }

?>
</table>
<form method="post" action="EditInfo.php">
    <input type="hidden" name="username" value = "<?php echo $username1;?>"/>
  <input type="hidden" name="password" value = "<?php echo $password1;?>"/>
  <button class="button" style="vertical-align:middle"><span>Edit Info </span></button>
</form>

    <a href="Login.html"><button class="button1" style="vertical-align:middle" style ="float: right;"><span>logout </span></button>


</body>

</html>