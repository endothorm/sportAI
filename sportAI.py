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

if classLabel == "air hockey":
        print("Air hockey is a Pong-like tabletop sport where two opposing players try to score goals against each other on a low-friction table using two hand-held discs (mallets) and a lightweight plastic puck. Source: Wikipedia")
elif classLabel == "ampute football":
        print("Amputee football is a variation on football played by people missing one or more limbs.")
elif classLabel == "archery":
        print("Archery is a sport played with bows and arrows to test athletes' aim, strength and skill with the bow.")
elif classLabel == "arm wrestling":
        print("Arm wrestling is a sport played, usually as a joke, involving attempting to flatten the other player's arm against a table.")
elif classLabel == "axe throwing":
        print("The modern sport of axe throwing involves a competitor throwing an axe at a target, attempting to hit the bullseye as near as possible. ")
elif classLabel == "balance beam":
        print("Balance beam is an event within gymnastics where gymnasts compete on a four-inch wide beam, performing jumps and flips.")
elif classLabel == "barell racing":
        print("Barrel racing is a rodeo event in which a horse and rider attempt to run a cloverleaf pattern around preset barrels in the fastest time.")
elif classLabel == "baseball":
        print("Baseball is a bat-and-ball sport played between two teams of nine players each, taking turns batting and fielding.")
elif classLabel == "basketball":
        print("Basketball is a game where two opposing teams attempt to send a ball through their opponent's hoop while preventing the opposite team from scoring.")
elif classLabel == "baton twirling":
        print("Baton twirling is a sport in which an athlete dances and twirls a baton slightly longer than their arm.")
elif classLabel == "bike polo":
        print("Bicycle polo is a team sport, similar to traditional polo, except that bicycles are used instead of horses. ")    
elif classLabel == "billiards":
        print("Billiards is a family of sports played on a pool table with billiard balls and cues.")
elif classLabel == "bmx":
        print("BMX is a cycle sport performed on BMX bikes, either in competitive BMX racing or freestyle BMX, or else in general street or off-road recreation.")
elif classLabel == "bobsled":
        print("Bobsleigh or bobsled is a winter sport in which teams of 2 to 4 athletes make timed runs down narrow, twisting, banked, iced tracks in a gravity-powered sleigh. ")
elif classLabel == "bowling":
        print("Bowling is a sport played in a bowling alley where the goal is to knock pins over with a heavy ball, rolled with the hands.")
elif classLabel == "boxing":
        print("Boxing is a combat sport and martial art. Taking place in a boxing ring, it involves two people  usually wearing protective equipment, such as protective gloves, hand wraps, and mouthguards  throwing punches at each other for a predetermined amount of time.")    
elif classLabel == "bull riding":
        print("Bull riding is a rodeo sport that involves a rider getting on a bucking bull and attempting to stay mounted while the animal tries to buck off the rider.")
elif classLabel == "bungee jumping":
        print("Bungee jumping, also spelled bungy jumping, is an activity that involves a person jumping from a great height while connected to a large elastic cord.")   
elif classLabel == "canoe slamon":
        print("Canoe slalom is a competitive sport in which an athlete steers a canoe through a series of gates in a river, trying for the best time.")
elif classLabel == "cheerleading":
        print("Cheerleading is an activity in which the participants (called cheerleaders) cheer for their team as a form of encouragement.")   
elif classLabel == "chuckwagon racing":
        print("Chuckwagon racing is an equestrian rodeo sport in which drivers in a chuckwagon led by a team of Thoroughbred horses race around a track.")
elif classLabel == "cricket":
        print("Cricket is a bat-and-ball game that is played between two teams of eleven players on a field, at the centre of which is a 22-yard (20-metre) pitch with a wicket at each end, each comprising two bails balanced on three stumps.")
elif classLabel == "croquet":
        print("Croquet is a sport played with balls and mallets, the object of which is to get a small ball through a wicket.")
elif classLabel == "curling":
        print("Curling is a sport in which players slide stones on a sheet of ice toward a target area which is segmented into four concentric circles.")
elif classLabel == "disc golf":
        print("Disc golf, also known as frisbee golf, is a flying disc sport in which players throw a disc at a target; it is played using rules similar to golf.")
elif classLabel == "fencing":
        print("Fencing is a combat sport that features sword fighting.")
elif classLabel == "field hockey":
        print("Field hockey (or simply hockey) is a team sport structured in standard hockey format.")
elif classLabel in [ "figure skating men", "figure skating pairs", "figure skating women"]:
        print("Figure skating is a sport in which individuals, pairs, or groups perform on figure skates on ice.")
elif classLabel == "fly fishing":
        print("Fly fishing is an activity in which participants attempt to catch fish with rods and fly lures.")
elif classLabel == "football":
        print("American football, referred to simply as football in the United States and Canada and also known as gridiron football, is a team sport played by two teams of eleven players on a rectangular field with goalposts at each end.")
elif classLabel == "formula 1 racing":
        print("Formula one racing is a sport in which extremely fast cars are driven repeatedly around a track.")
elif classLabel == "frisbee":
        print("Flying disc sports are sports or games played with discs, often called by the trademarked name Frisbees. Ultimate and disc golf are sports with substantial international followings.")
elif classLabel == "gaga":
        print("Gaga (Hebrew: גע גע literally 'touch touch') (also: ga-ga, gaga ball, or ga-ga ball) is a variant of dodgeball that is played in a gaga pit.")
elif classLabel == "giant slalom":
        print("Giant slalom (GS) is an alpine skiing and alpine snowboarding competitive discipline. It involves racing between sets of poles (gates) spaced at a greater distance from each other than in slalom but less than in Super-G.")
elif classLabel == "golf":
        print("Golf is a sport played with clubs and a small ball on a golf course, in which the ball is struck towards a hole.")
elif classLabel == "hammer throw":
        print("The hammer throw is one of the four throwing events in regular outdoor track and field competitions, along with the discus throw, shot put and javelin.")
elif classLabel == "hang gliding":
        print("Hang gliding is an air sport or recreational activity in which a pilot flies a light, non-motorised, heavier-than-air aircraft called a hang glider.")
elif classLabel == "harness racing":
        print("Harness racing is a form of horse racing in which the horses race at a specific gait (a trot or a pace). They usually pull a two-wheeled cart called a sulky, spider, or chariot occupied by a driver.")
elif classLabel == "high jump":
        print("The high jump is a track and field event in which competitors must jump unaided over a horizontal bar placed at measured heights without dislodging it.")
elif classLabel == "hockey":
        print("Hockey is an ice sport played with clubs and a disc-shaped puck on skates.")
elif classLabel == "horse jumping":
        print("Horse jumping is a competitive sport where riders attempt to jump their horses over varying heights of hurdle.")
elif classLabel == "horse racing":
        print("Horse racing is a competitive sport in which a jockey rides a horse for speed and endurance.")
elif classLabel == "horseshoe pitching":
        print("Horseshoes is a lawn game played between two people (or two teams of two people) using four horseshoes and two throwing targets (stakes) set in a lawn or sandbox area. ")
elif classLabel == "hurdles":
        print("Hurdling is the act of jumping over an obstacle at a high speed or in a sprint.")
elif classLabel == "hydroplane racing":
        print("Hydroplane racing (also known as hydro racing) is a sport involving racing hydroplanes on lakes, rivers, and bays. It is a popular spectator sport in several countries.")
