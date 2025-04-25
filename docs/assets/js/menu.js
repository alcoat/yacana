// Menu initialization function
function initializeMenu(templatePath = 'templates/tuto_nav.html') {
    // Load the menu template
    fetch(templatePath)
        .then(response => response.text())
        .then(html => {
            document.getElementById('menu-container').innerHTML = html;
            
            // Initialize menu event handlers after template is loaded
            var $menu = $('.menu'),
                $menu_openers = $menu.children('ul').find('.opener');

            $menu_openers.each(function() {
                var $this = $(this);

                $this.on('click', function(event) {
                    // Prevent default.
                    event.preventDefault();

                    // Toggle.
                    $menu_openers.not($this).removeClass('active');
                    $this.toggleClass('active');

                    // Trigger resize (sidebar lock).
                    $(window).triggerHandler('resize.sidebar-lock');
                });
            });
        });
} 