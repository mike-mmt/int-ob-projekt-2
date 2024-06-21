function(e) {
  var FuzzySet = function (p1, p2, p3) {
    "use strict";
    return {
      className: "FuzzySet",
      getArea: function () {
        if (this.totalarea === undefined) {
          if (p3 > p2 && p2 > p1) {
            this.totalarea = (p3 - p1) / 2;
          } else {
            this.totalarea = 0;
          }
        }
        return this.totalarea;
      },
      getCenter: function () {
        return p2;
      },
      getFuzzyArea: function () {
        return this.weighedArea || 0;
      },
      calculateFuzzyArea: function () {
        //a private function
        this.weighedArea =
          (1 - Math.pow(1 - this.getFuzzyValue(), 2)) * this.getArea();
      },
      getFuzzyValue: function () {
        return this.fuzzyValue || 0;
      },
      setFuzzyValue: function (value) {
        this.fuzzyValue = value;
        this.calculateFuzzyArea();
      },
      calculateFuzzyValue: function (x) {
        this.fuzzyValue = 0;
        this.crispValue = x;
        if (p3 > p2 && p2 > p1) {
          if (x > p1 && x <= p2) {
            this.fuzzyValue = (x - p1) / (p2 - p1);
          } else if (x > p2 && x < p3) {
            this.fuzzyValue = 1.0 - (x - p2) / (p3 - p2);
          }
        }
        this.calculateFuzzyArea();
        return this.fuzzyValue;
      },
    };
  };

  /*
FuzzyVariable defines a list of continious fuzzyset over some  range of crips values
The constructor does not take any parameters

Object properties and methods
------------------------------------
getFuzzySets : returns the list of fuzzy sets
fuzzify(x) :   calculates the fuzzy set values given crisp value x and returns the list of the fuzzysets
defuzzify() :  given a list of fuzzyset values calculates the crisp value x
fireRules() :  loopls through all sets and check if rule is assigned and executes it to update the set value
 */

  var FuzzyVariable = function () {
    "use strict";
    var that = this;
    return {
      className: "FuzzyVariable",
      fuzzyfy: function (v) {
        var fuzzysets = this.getFuzzySets(),
          n = fuzzysets.length,
          i = 0;
        for (i = 0; i < n; i += 1) {
          fuzzysets[i].variable = that;
          fuzzysets[i].calculateFuzzyValue(v);
        }
      },
      defuzzify: function () {
        //uses centroid method to calculate the crisp fuzzy value
        //calculates the weighed centroid of fuzzy values ie
        // y = sum(m*c)/N  where m is the fuzzy value (weight) and c is the center of the
        // set  (p2) and N is the number of sets calulated
        //the weight is a function of the fuzzy set value
        var fuzzysets = this.getFuzzySets(),
          n = fuzzysets.length,
          i = 0,
          sumOfWeights = 0,
          weighedSum = 0;

        for (i = 0; i < n; i += 1) {
          weighedSum += fuzzysets[i].getFuzzyArea() * fuzzysets[i].getCenter();
          sumOfWeights += fuzzysets[i].getFuzzyArea();
        }
        return sumOfWeights === 0 ? 0 : weighedSum / sumOfWeights;
      },
      getFuzzySets: function () {
        var items = [],
          property;
        for (property in this) {
          if (
            this.hasOwnProperty(property) &&
            this[property].className === "FuzzySet"
          ) {
            this[property].name = property;
            items.push(this[property]);
          }
        }
        return items;
      },
      fireRules: function () {
        var fuzzysets = this.getFuzzySets(),
          n = fuzzysets.length,
          fuzzyset,
          rule,
          i = 0;
        for (i = 0; i < n; i += 1) {
          fuzzyset = fuzzysets[i];
          rule = fuzzyset.rule;
          //if set has rule then calculate the output value
          if (rule !== undefined && rule.className === "FuzzyRule") {
            fuzzyset.setFuzzyValue(rule.fire());
          }
        }
      },
    };
  };

  // Represents a AND expression of the form
  // exp [AND exp]*
  // Methods:
  // fire() : returns the expression fuzzy value
  var FuzzyRule = function () {
    "use strict";
    return {
      className: "FuzzyRule",
      addExpr: function (expr) {
        //adds a list of fuzzysets that forms a AND expression
        //the list represents ORed AND expressions
        if (this.list === undefined) {
          this.list = [];
        }
        this.list.push(expr);
      },
      fire: function () {
        var i,
          j,
          n,
          m,
          min,
          max,
          set,
          fuzzyValue = 0;

        if (this.list === undefined) {
          return 0;
        }
        min = 1;
        max = 0;
        n = this.list.length;
        //loop through the or expressions and find the maximum fuzzy value from each and expression
        for (i = 0; i < n; i += 1) {
          m = this.list[i].length;
          min = 1;
          //loop through and expression to get it's min value
          for (j = 0; j < m; j += 1) {
            set = this.list[i][j];
            if (
              set !== undefined &&
              set.className === "FuzzySet" &&
              set.value !== "undefined"
            ) {
              fuzzyValue = set.getFuzzyValue();
              if (fuzzyValue < min) {
                min = fuzzyValue;
              }
            }
          }
          if (min > max) {
            max = min;
          }
        }
        return max;
      },
    };
  };

  function calculate_angle(x1, y1, x2, y2) {
    return (360 + (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI) % 360;
  }

  function calculate_angle_difference(desired_angle, tank_rotation) {
    return ((desired_angle - tank_rotation + 180) % 360) - 180;
  }

  function updateDirectionResponse(response, n) {
    switch (n) {
      case 1:
        response.turnLeft = 1;
        break;
      case -1:
        response.turnRight = 1;
        break;
      default:
        break;
    }
    return response;
  }

  function updateMoveResponse(response, n) {
    switch (n) {
      case 1:
        response.goForward = 1;
        break;
      case -1:
        response.goBack = 1;
        break;
      default:
        break;
    }
    return response;
  }

  function updateCannonResponse(response, n) {
    switch (n) {
      case 1:
        response.cannonLeft = 1;
        break;
      case -1:
        response.cannonRight = 1;
        break;
      default:
        break;
    }
    return response;
  }

  var response = {};

  angle = FuzzyVariable();
  // angle.VL = FuzzySet(-270, -180, -90);
  angle.L = FuzzySet(-330, -180, -30);
  angle.M = FuzzySet(-30, 0, 30);
  angle.R = FuzzySet(30, 180, 330);
  // angle.VR = FuzzySet(90, 180, 270);

  posx = FuzzyVariable();
  posx.L = FuzzySet(-200, 0, 200);
  posx.C = FuzzySet(200, 250, 300);
  posx.H = FuzzySet(300, 500, 700);

  posy = FuzzyVariable();
  posy.L = FuzzySet(-200, 0, 200);
  posy.C = FuzzySet(200, 250, 300);
  posy.H = FuzzySet(300, 500, 700);

  rotateOutput = FuzzyVariable();
  rotateOutput.L = FuzzySet(-2.0, -1.0, 0.0);
  rotateOutput.N = FuzzySet(-1.0, 0.0, 1.0);
  rotateOutput.R = FuzzySet(0.0, 1.0, 2.0);

  rotateOutput.L.rule = FuzzyRule();
  rotateOutput.L.rule.addExpr([angle.R]);

  rotateOutput.N.rule = FuzzyRule();
  rotateOutput.N.rule.addExpr([angle.M, posx.C, posy.C]);

  rotateOutput.R.rule = FuzzyRule();
  rotateOutput.R.rule.addExpr([angle.L]);

  moveOutput = FuzzyVariable();

  //   moveOutput.B = FuzzySet(-2.0, -1.0, 0.0);
  moveOutput.N = FuzzySet(-1.0, 0.0, 1.0);
  moveOutput.F = FuzzySet(0.0, 1.0, 2.0);

  // moveOutput.B.rule = FuzzyRule();
  // moveOutput.B.rule.addExpr([]);

  moveOutput.N.rule = FuzzyRule();
  moveOutput.N.rule.addExpr([posx.C, posy.C]);

  moveOutput.F.rule = FuzzyRule();
  moveOutput.F.rule.addExpr([posx.L]);
  moveOutput.F.rule.addExpr([posy.L]);
  moveOutput.F.rule.addExpr([posx.H]);
  moveOutput.F.rule.addExpr([posy.H]);

  cannonAngle = FuzzyVariable();
  cannonAngle.L = FuzzySet(-350, -180, -2.5);
  cannonAngle.M = FuzzySet(-2.5, 0, 2.5);
  cannonAngle.R = FuzzySet(2.5, 180, 350);

  rotateCannonOutput = FuzzyVariable();
  rotateCannonOutput.L = FuzzySet(-2.0, -1.0, 0.0);
  rotateCannonOutput.N = FuzzySet(-1.0, 0.0, 1.0);
  rotateCannonOutput.R = FuzzySet(0.0, 1.0, 2.0);


  rotateCannonOutput.L.rule = FuzzyRule();
    rotateCannonOutput.L.rule.addExpr([cannonAngle.R]);

  rotateCannonOutput.N.rule = FuzzyRule();
    rotateCannonOutput.N.rule.addExpr([cannonAngle.M]);

    rotateCannonOutput.R.rule = FuzzyRule();
    rotateCannonOutput.R.rule.addExpr([cannonAngle.L]);


  var tank = e.data.myTank;

  var desired_angle = calculate_angle(tank.x, tank.y, 250, 250);
  var angle_diff = calculate_angle_difference(desired_angle, tank.rotation);

  var desired_cannon_angle = calculate_angle(tank.x, tank.y, e.data.enemyTank.x, e.data.enemyTank.y);
  var cannon_angle_diff = calculate_angle_difference(desired_cannon_angle, tank.cannonRotation);

  console.log("Desired: ", desired_cannon_angle, "Diff: ", cannon_angle_diff)

  angle.fuzzyfy(angle_diff);
  posx.fuzzyfy(tank.x);
  posy.fuzzyfy(tank.y);
  cannonAngle.fuzzyfy(cannon_angle_diff);

  rotateOutput.fireRules();
  moveOutput.fireRules();
  rotateCannonOutput.fireRules();

  var rotationActionValue = rotateOutput.defuzzify();
  var moveActionValue = moveOutput.defuzzify();
  var cannonRotationActionValue = rotateCannonOutput.defuzzify();
  // response.goForward = 1;
//   console.log(rotationActionValue, moveActionValue, cannonRotationActionValue);
  response = updateDirectionResponse(response, rotationActionValue);
  response = updateMoveResponse(response, moveActionValue);
  response = updateCannonResponse(response, cannonRotationActionValue);
  response.shoot = 1;
//   console.log(response);
  self.postMessage(response);
}
