// Function to display the uploaded image before submitting the form
function previewImage(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            $('#image-preview').attr('src', e.target.result);
        }
        reader.readAsDataURL(input.files[0]);
    }
}

// Function to submit the form on file selection
$(document).ready(function() {
    $('#file-input').change(function() {
        previewImage(this);
        $('#upload-form').submit();
    });
});
