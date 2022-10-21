// default seattle coords
// centered tableData coords
const seattle = new google.maps.LatLng(47.61922313, -122.3357037);

// create map, center on Seattle
const map = new google.maps.Map(document.getElementById("map"), {
  zoom: 11,
  center: seattle,
});

// create empty list for markers 
markers = []

// create marker
function createMarker(row){
  return new google.maps.Marker({
    position: {
      lat: parseFloat(row.latitude), 
      lng: parseFloat(row.longitude)
    }
  });
}

// add marker
function addMarkers(markers){
  for (let marker of markers){
    marker.setMap(map);
  }
}

// clear markers
function removeMarkers(markers){
  for (let marker of markers){
    marker.setMap(null);
  }
}
