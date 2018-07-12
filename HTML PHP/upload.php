<!DOCTYPE html>

<?php

 $username1 = trim($_REQUEST['username']);
 $password1 = trim($_REQUEST['password']);
 $fname = $_REQUEST["fname"];
 $lname = $_REQUEST["lname"]; 
 $dob = $_REQUEST["dob"];
 $bg = $_REQUEST["bg"];
 $cno = $_REQUEST["cno"];


$conn = mysqli_connect("localhost", "id5401488_kurien", "123qwe", "id5401488_app_data");
  // Check connection
  if ($conn->connect_error) {
   die("Connection failed: " . $conn->connect_error);
  } 

  $sqlUp =  "UPDATE PerfitLogin SET FirstName ='".$fname."',LastName ='".$lname."',DoB ='".$dob."', BloodGroup = '".$bg."',ContactNo ='".$cno."' WHERE username = '".$username1."'";
    $resultUp = $conn->query($sqlUp);

 include 'Details.php';
 ?>
<html>
    <input type="hidden" name="username" value = "<?php echo $username1;?>"/>
	<input type="hidden" name="password" value = "<?php echo $password1;?>"/>
</html>
