phone_width = 71.6;
phone_height = 147.6;
phone_depth = 7.8;

wall_thickness = 2.5;
tolerance = 2;
cradle_depth = 12;

base_diameter = 65;
base_height = 18;
suction_rim_height = 8;

arm_thickness = 18;
arm_width = 22;
joint_ball_diameter = 20;



module pritrdi_na_avto() {
translate([0, 0, 0]) {
cylinder(h=base_height*0.6, d1=base_diameter, d2=base_diameter*0.8, $fn=64);
//zgornja
translate([0, 0, base_height*0.6])
cylinder(h=base_height*0.4, d=base_diameter*0.8, $fn=64);
}
    
translate([0, 0, -suction_rim_height])
difference() {
cylinder(h=suction_rim_height, d=base_diameter, $fn=64);
translate([0, 0, -1])
cylinder(h=suction_rim_height-2, d=base_diameter-4, $fn=64);
}
    
//dam skuoaj
translate([0, base_diameter*0.3, base_height*0.7])
rotate([45, 0, 0])
cylinder(h=25, d=joint_ball_diameter+4, $fn=32);
}



module rocaj() {
hull() {
translate([0, 0, 0])
rotate([0, 90, 0])
cylinder(h=arm_thickness, d=arm_width, $fn=32);
        
//srednji del
translate([40, 25, 15])
rotate([0, 90, 0])
cylinder(h=arm_thickness, d=arm_width, $fn=32);
        
//koncni del
translate([80, 15, 25])
rotate([0, 90, 0])
cylinder(h=arm_thickness, d=arm_width, $fn=32);
    }
    
//zdruzim na zacetku
translate([arm_thickness/2, 0, 0])
sphere(d=joint_ball_diameter, $fn=32);
    
translate([80 + arm_thickness/2, 15, 25])
sphere(d=joint_ball_diameter, $fn=32);
}






module drzalo_za_telefon() {
hull() {
minkowski(){
cube([phone_width + tolerance + 2*wall_thickness, cradle_depth, phone_height*0.7 + wall_thickness]);
sphere(r=2.0, $fn=30);
}
           
}

//drzali na levi in desni
translate([-8, 0, phone_height*0.3]) {
minkowski(){
difference() {
//trapezoid
        linear_extrude(height = phone_height * 0.25) {
            polygon(points = [
                [0, 0],                             //spodaj levo
                [8, 0],                             // spodaj desno
                [8 - 3, cradle_depth + 15],         // zgoraj desno
                [0 + 3, cradle_depth + 15]          // spodaj levo
            ]);
        }

        for (i = [0:3]) {
            translate([-1, cradle_depth/2 + 5, 5 + i*15])
                rotate([0, 90, 0])
                    cylinder(h=10, d=2, $fn=16);
        }
    }
            sphere(r=1, $fn=16);
    }
}


translate([phone_width + tolerance + 2*wall_thickness, 0, phone_height*0.3]) {
minkowski() {
difference() {
            linear_extrude(height = phone_height * 0.25) {
                polygon(points = [
                    [0, 0],                          
                    [8, 0],                            
                    [8 - 3, cradle_depth + 15],        
                    [0 + 3, cradle_depth + 15]          
                ]);
            }


            for(i = [0:3]) {
                translate([-1, cradle_depth/2 + 5, 5 + i*15])
                    rotate([0, 90, 0])
                        cylinder(h=10, d=2, $fn=16);
            }
        }
        sphere(r=1, $fn=16);
    }
}

//spodnja drzala 
for (x_shift = [15, phone_width + tolerance + 2*wall_thickness - 15]) {
      translate([x_shift, cradle_depth/2 + 10, -5]) {
          rotate([0, 0, 90])
          scale([1.5, 1, 0.5])  
              minkowski() {
                  cylinder(h = 6, r = 3.5, $fn=32); 
                  sphere(r = 5.0, $fn=34);      
                }
        }
    }


//zadaj da prikljucim
translate([phone_width/2 + wall_thickness, cradle_depth - 10, phone_height*0.35])
rotate([90, 0, 0])
cylinder(h=15, d=joint_ball_diameter+6, $fn=32);
}







// Complete assembly
module celotno_drzalo_za_telefon() {
    // Suction cup base
    color("gray"){
    pritrdi_na_avto();
    
    // Curved arm
    translate([0, base_diameter*0.3, base_height*0.7])
        rotate([45, 0, 0])
            translate([12, 0, 0])
                rocaj();
    
    // Phone cradle
 translate([60, base_diameter*0.3 + 50, base_height*0.7 + 5])
    rotate([45, 0, 0])
        drzalo_za_telefon();
   
}
}


celotno_drzalo_za_telefon();
