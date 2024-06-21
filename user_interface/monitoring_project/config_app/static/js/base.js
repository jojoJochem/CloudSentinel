function showOverlay() {
    document.getElementById('overlay').style.display = 'block';
}

function hideOverlay() {
    document.getElementById('overlay').style.display = 'none';
}

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('uploadForm');
    if (form) {
        form.addEventListener('submit', function () {
            showOverlay();
        });
    }
});

document.addEventListener('DOMContentLoaded', function () {
    var dropdownElements = document.querySelectorAll('.dropdown-submenu');
    dropdownElements.forEach(function (element) {
        element.addEventListener('mouseover', function (e) {
            var submenu = e.currentTarget.querySelector('.dropdown-menu');
            var rect = submenu.getBoundingClientRect();
            submenu.style.left = -rect.width + 'px';
            submenu.classList.add('show');
        });
        element.addEventListener('mouseout', function (e) {
            var submenu = e.currentTarget.querySelector('.dropdown-menu');
            submenu.classList.remove('show');
        });
    });
});
