#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(model="resnet18.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")

class_idx, confidence = net.Classify(img)
classLabel = net.GetClassDesc(class_idx)
print("image is  "+ str(classLabel) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")

sports_blurb = ["Air hockey is a Pong-like tabletop sport where two opposing players try to score goals against each other on a low-friction table using two hand-held discs (mallets) and a lightweight plastic puck. Source: Wikipedia", "Amputee football is a variation on football played by people missing one or more limbs.",
                "Archery is a sport played with bows and arrows to test athletes' aim, strength and skill with the bow.", "Arm wrestling is a sport played, usually as a joke, involving attempting to flatten the other player's arm against a table."
                 , "The modern sport of axe throwing involves a competitor throwing an axe at a target, attempting to hit the bullseye as near as possible. ",  "Balance beam is an event within gymnastics where gymnasts compete on a four-inch wide beam, performing jumps and flips.",
                "Barrel racing is a rodeo event in which a horse and rider attempt to run a cloverleaf pattern around preset barrels in the fastest time.",  "Baseball is a bat-and-ball sport played between two teams of nine players each, taking turns batting and fielding."
                ,"Baton twirling is a sport in which an athlete dances and twirls a baton slightly longer than their arm.", "Bicycle polo is a team sport, similar to traditional polo, except that bicycles are used instead of horses. ","Billiards is a family of sports played on a pool table with billiard balls and cues.",
                 "BMX is a cycle sport performed on BMX bikes, either in competitive BMX racing or freestyle BMX, or else in general street or off-road recreation.",
                 "Bobsleigh or bobsled is a winter sport in which teams of 2 to 4 athletes make timed runs down narrow, twisting, banked, iced tracks in a gravity-powered sleigh. ",
                "Bowling is a sport played in a bowling alley where the goal is to knock pins over with a heavy ball, rolled with the hands.",   "Boxing is a combat sport and martial art. Taking place in a boxing ring, it involves two people throwing punches at each other for a predetermined amount of time.",
                 "Bull riding is a rodeo sport that involves a rider getting on a bucking bull and attempting to stay mounted while the animal tries to buck off the rider.", 
                  "Bungee jumping, also spelled bungy jumping, is an activity that involves a person jumping from a great height while connected to a large elastic cord.",
                         "Canoe slalom is a competitive sport in which an athlete steers a canoe through a series of gates in a river, trying for the best time.",
                                "Cheerleading is an activity in which the participants (called cheerleaders) cheer for their team as a form of encouragement.",
                                  "Chuckwagon racing is an equestrian rodeo sport in which drivers in a chuckwagon led by a team of Thoroughbred horses race around a track.",
                                    "Cricket is a bat-and-ball game that is played between two teams of eleven players on a field, at the centre of which is a 22-yard (20-metre) pitch with a wicket at each end, each comprising two bails balanced on three stumps.",
                                 "Croquet is a sport played with balls and mallets, the object of which is to get a small ball through a wicket.",
                                   "Curling is a sport in which players slide stones on a sheet of ice toward a target area which is segmented into four concentric circles.",
                                      "Disc golf, also known as frisbee golf, is a flying disc sport in which players throw a disc at a target; it is played using rules similar to golf.",
                                       "Fencing is a combat sport that features sword fighting.",   "Field hockey (or simply hockey) is a team sport structured in standard hockey format.", "Figure skating is a sport in which individuals, pairs, or groups perform on figure skates on ice.",
                                       "Figure skating is a sport in which individuals, pairs, or groups perform on figure skates on ice.",
                                       "Figure skating is a sport in which individuals, pairs, or groups perform on figure skates on ice.",
                                         "Fly fishing is an activity in which participants attempt to catch fish with rods and fly lures.",
                                          "American football, referred to simply as football in the United States and Canada and also known as gridiron football, is a team sport played by two teams of eleven players on a rectangular field with goalposts at each end.",
                                            "Formula one racing is a sport in which extremely fast cars are driven repeatedly around a track.", 
                                              "Flying disc sports are sports or games played with discs, often called by the trademarked name Frisbees. Ultimate and disc golf are sports with substantial international followings.",
                                                  "Gaga (Hebrew: גע גע literally 'touch touch') (also: ga-ga, gaga ball, or ga-ga ball) is a variant of dodgeball that is played in a gaga pit.",
                                                     "Giant slalom (GS) is an alpine skiing and alpine snowboarding competitive discipline. It involves racing between sets of poles (gates) spaced at a greater distance from each other than in slalom but less than in Super-G.",
                                                        "Golf is a sport played with clubs and a small ball on a golf course, in which the ball is struck towards a hole.",
                                                              "The hammer throw is one of the four throwing events in regular outdoor track and field competitions, along with the discus throw, shot put and javelin.",
                                                                  "Hang gliding is an air sport or recreational activity in which a pilot flies a light, non-motorised, heavier-than-air aircraft called a hang glider.",
                                                                    "Harness racing is a form of horse racing in which the horses race at a specific gait (a trot or a pace). They usually pull a two-wheeled cart called a sulky, spider, or chariot occupied by a driver.",
                                                                        "The high jump is a track and field event in which competitors must jump unaided over a horizontal bar placed at measured heights without dislodging it.",
                                                                                 "Hockey is an ice sport played with clubs and a disc-shaped puck on skates.",
                                                                                       "Horse jumping is a competitive sport where riders attempt to jump their horses over varying heights of hurdle.",
                        "Horse racing is a competitive sport in which a jockey rides a horse for speed and endurance.",
                "Horseshoes is a lawn game played between two people (or two teams of two people) using four horseshoes and two throwing targets (stakes) set in a lawn or sandbox area. ",
                "Hurdling is the act of jumping over an obstacle at a high speed or in a sprint.",
                "Hydroplane racing (also known as hydro racing) is a sport involving racing hydroplanes on lakes, rivers, and bays. It is a popular spectator sport in several countries.",
"Ice climbing is a sport in which athletes climb a wall of ice using picks and other equipment.",
"Ice yachting is a sport involving a craft called an ice yacht, essentially a sled with sails.",
"Jai alai is a sport involving bouncing a ball off a walled-in space by accelerating it to high speeds with a hand-held wicker, commonly referred to as a cesta",
"The javelin throw is a track and field event where the javelin, a spear about 2.5 m (8 ft 2 in) in length, is thrown as far as possible. ",
"Jousting is a sport in which two participants, mounted on horseback, ride at each other and attempt to unseat each other with lances.",
"Judo is an unarmed modern Japanese martial art, combat sport, Olympic sport (since 1964), and the most prominent form of jacket wrestling competed internationally",
"Lacrosse is a contact team sport played with a lacrosse stick and a lacrosse ball.", "Log rolling, sometimes called birling, is a Sparring Sport involving two competitors, each on one end of a free-floating log in a body of water.", "A luge is a small one- or two-person sled on which one sleds supine (face-up) and feet-first. Luge is the name of the olympic sport that employs that kind of sled.",
"The motorcycle sport of racing (also called moto racing and motorbike racing) includes motorcycle road racing and off-road racing, both either on circuits or open courses, and track racing.",
"Mushing is a sport or transport method powered by dogs.", "Nascar racing is a form of car racing", "Wrestling is a sport in which two opponents attempt to throw the other off balance", 
"The parallel bar is a subsection of men's gymnastics with two parallel bars used for flips and handstands", "Pole climbing is ascending a pole which one can grip with their hands. ",
"Pole dance combines dance and acrobatics centered around a vertical pole.", "Pole vaulting is a sport in which athletes attempt to flip themselves into the air and over a high hurdle with a long pole", 
"Polo is a ball game that is played on horseback, a traditional field sport and one of the world's oldest known team sports.", "The pommel horse is an artistic gymnastics event held at the Summer Olympics. ",
"The rings, also known as still rings[1] (in contrast to flying rings), is an artistic gymnastics apparatus and the event that uses it.",
"Rock climbing is a sport in which a sheer rock wall with handholds, natural or unnatural, is scaled", "Roller derby is a ballgame played on roller skates.",
"Aggressive inline skating (referred to by participants as rollerblading, blading, skating or rolling) is a sub-discipline primarily focused on the execution of tricks in the action sports canon. ",
"Rowing is a water sport where athletes attempt to beat competitors on a long boat with oars", "Rugby football is the collective name for the team sports of rugby union or rugby league.", "Sail racing is sailing a small wind-powered boat for speed.",
"The shot put is a track and field event involving throwing a heavy spherical ball—the shot—as far as possible.", "Shuffleboard is a game in which players use cues to push weighted discs, sending them gliding down a narrow court, with the purpose of having them come to rest within a marked scoring area. ", 
"Sidecar racing is a car-racing sport on a track", "Ski jumping is a sport in which athletes ride off jumps and attempt to execute difficult tricks on skis.", "Sky surfing is a type of skydiving and extreme sport in which the skydiver wears a custom skysurf board attached to the feet and performs surfing-style aerobatics during freefall.",
"Skydiving is an aerial sport in which athletes jump out of planes.", "Snowboarding is a recreational and competitive activity that involves descending a snow-covered surface while standing on a snowboard that is almost always attached to a rider's feet. ",
"A snowmobile, also known as a snowmachine, motor sled, motor sledge, skimobile, or snow scooter, is a motorized vehicle designed for winter travel and recreation on snow. It is raced within the sport of snowmobile racing.",
"Speed skating is a competitive form of ice skating in which the competitors race each other in travelling a certain distance on skates. ",
"Steer wrestling, also known as bulldogging, is a rodeo event in which a horse-mounted rider chases a steer, drops from the horse to the steer, then wrestles the steer to the ground by grabbing its horns and pulling it off-balance so that it falls to the ground. ",
"Sumo is a form of competitive full-contact wrestling where a rikishi (wrestler) attempts to force his opponent out of a circular ring (dohyō) or into touching the ground with any body part other than the soles of his feet (usually by throwing, shoving or pushing him down).",
"Surfing is a sport in which an athlete rides waves on a flat, rigid board.",
"Swimming is a family of aquatic sports and races.", 
"Table tennis (also known as ping-pong or whiff-whaff) is a racket sport derived from tennis but distinguished by its playing surface being atop a stationary table, rather than the court on which players stand. ",
"Tennis is a racket sport played on a court.","Track cycling is a sport in which a oval-shaped track is repeatedly ridden around on bicycles.",
"Trapeze is a circus sport involving a long swing, the trapeze.", "Tug of war is a sport that pits two teams against each other in a test of strength: teams pull on opposite ends of a rope, with the goal being to bring the rope a certain distance in one direction against the force of the opposing team's pull.",
"Ultimate is a variation frisbee sport.", "Uneven bars is a women's gymnastics event involving doing flips around two bars positioned at different heights.",
"Volleyball is a team sport in which two teams of six players are separated by a net. ", 
"Water cycling is a sport involving pedal-powered aquatic vehicles.",
"Water polo is a competitive team sport played in water between two teams of seven players each. "
, "Weightlifting is a family of sports involving weights of varoious masses.", "Wheelchair basketball is a style of basketball played using a sports wheelchair.",
"Wheelchair racing is a para sport in which sport wheelchairs are raced along a track.",
"Wingsuit flying (or wingsuiting) is the sport of skydiving using a webbing-sleeved jumpsuit called a wingsuit to add webbed area to the diver's body and generate increased lift, which allows extended air time by gliding flight rather than just free falling. "]

if class_idx > 25:
        print(sports_blurb[class_idx - 1])
else:
        print(sports_blurb[class_idx])



