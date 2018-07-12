<!DOCTYPE html>

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
      while($row = $result->fetch_assoc()) {
  $fname = $row["FirstName"];
  $lname = $row["LastName"]; 
  $dob = $row["DoB"];
  $bg = $row["BloodGroup"];
  $cno = $row["ContactNo"];
      }
      
function Upload()
{
    $sqlUp =  "UPDATE PerfitLogin SET FirstName ='".$fname."',LastName ='".$lname."'WHERE username = '".$username1."'";
    $resultUp = $conn->query($sqlUp);
  if ($resultUp->num_rows > 0) {
      echo "Success";
  }
}

}
else { echo "0 results"; }
?>
<style>
    .button {
  display: inline-block;
  padding: 15px 25px;
  font-size: 24px;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
  outline: none;
  color: #fff;
  background-color: #4CAF50;
  border: none;
  border-radius: 15px;
  box-shadow: 0 9px #999;
}

.button:hover {background-color: #3e8e41}

.button:active {
  background-color: #3e8e41;
  box-shadow: 0 5px #666;
  transform: translateY(4px);
}
</style>
<head>
	<title>Login</title>
	<link rel="stylesheet" href="style.css">
</head>
<body>
	<section class="editInfo">
		<form name="Edit" action="upload.php"method ="post"  accept-charset="utf-8">
			<ul>
				<li>
					<label for="fname">First Name</label>
					<input type="text" name="fname" value = "<?php echo $fname;?>"><br/>
				</li>
				<li>
					<label for="lname">Last Name</label>
					<input type="text" name="lname" value = "<?php echo $lname;?>"><br/>
				</li>
				<li>
					<label for="dob">DoB</label>
					<input type="date" name="dob" value = "<?php echo $dob;?>"><br/>
				</li>
				<li>
					<label for="bg">Blood Group</label>
					<input type="text" name="bg" value = "<?php echo $bg;?>"><br/>
				</li>
				<li>
					<label for="cno">Contact No</label>
					<input type="number" name="cno" value = "<?php echo $cno;?>"><br/>
				</li>
				
					<input type="hidden" name="username" value = "<?php echo $username1;?>">
				
			
					<input type="hidden" name="password" value = "<?php echo $password1;?>">
			
				
			</ul>
			<button class="button">Submit</button>
		</form>
	</section>
</body>
</html>