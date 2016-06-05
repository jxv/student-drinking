var fs = require('fs');
var students = JSON.parse(fs.readFileSync('./students.json', 'utf8'));

var cvt = function(val) {
  if (val == 'yes') return true;
  if (val == 'no') return false;
  return val;
}

var xformDatum = function(attributes, datum) {
  var obj = {};
  for (var i = 0; i < datum.length; i++) {
    obj[attributes[i].name] = cvt(datum[i]);
  }
  return obj;
};

var xformDataSet = function(dataSet) {
  return dataSet.data.map(function(datum) { return xformDatum(dataSet.header.attributes, datum.values); });
};

console.log(JSON.stringify(xformDataSet(students)));
