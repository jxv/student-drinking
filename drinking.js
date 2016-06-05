var raw_students = require('./dataset/students_with_keys.json');

function binary(cond) {
  return cond ? 1 : 0;
}

function toNum(string) {
  return +string;
}

function refined(students) {
  return students.map(function(student) {
    return {
      "age": toNum(student.age),
      "famsize": binary(student.famsize == "GT3"),
      "internet": binary(student.internet),
      "activities": binary(student.activities),
      "sex": binary(student.sex == "M"),
      "walc": binary(student.Walc == '4' || student.Walc == '5')
    };
  });
}

var students = refined(raw_students);


function probGiven(examples, x) {
  var count = 0;
  var total = 0;
  examples.forEach(function (example) {
    if (example.x == x) {
      total += example.y;
      count++;
    }
  });
  return total / count;
}

function studentToExample(x, y) {
  return function(student){
    return {x: student[x], y: student[y]};
  };
}

var examples = students.map(studentToExample("internet", "walc"));

function allBinaryProbs(students) {
  students.map(function(student) {
    student.sex = cvtWithSex(student)
  });
}


function probWith(examples, valRange) {
  var probs = new Array(valRange);
  for (var i = 0; i < probs.length; i++) {
    probs[i] = probGiven(examples, i);
  }
  return probs;
}

console.log('internet: ', probWith(examples, 2));
