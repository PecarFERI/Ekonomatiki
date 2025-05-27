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
    

   
}
}


celotno_drzalo_za_telefon();