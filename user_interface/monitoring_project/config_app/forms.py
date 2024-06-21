from django import forms


def create_dynamic_form(config_data):
    class ConfigForm(forms.Form):
        pass

    for key, value in config_data.items():
        field_type = forms.CharField  # Default to CharField
        if isinstance(value, bool):
            field_type = forms.BooleanField
        elif isinstance(value, int):
            field_type = forms.IntegerField
        elif isinstance(value, float):
            field_type = forms.FloatField

        field = field_type(initial=value, required=False)
        ConfigForm.base_fields[key] = field

    return ConfigForm


class FileUploadForm(forms.Form):
    file = forms.FileField()


class MonitoringForm(forms.Form):
    containers = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=[], label='Containers', required=True, help_text="Select containers to monitor (should match the number of containers in the model)")
    model = forms.ChoiceField(choices=[], label='Model', required=True)
    data_interval = forms.IntegerField(min_value=1, required=True, initial=5, help_text="Measurement Interval in Seconds")
    duration = forms.IntegerField(min_value=10, required=True, initial=10, help_text="Test Duration in Minutes (> 30 minutes recommended)")
    test_interval = forms.FloatField(min_value=0.1, required=True, initial=5, help_text="Test Interval in Minutes (> 5 minutes recommended)")
    crca_threshold = forms.FloatField(min_value=0.0, required=True, initial=0.5, help_text="Threshold of Anomalies detected in Percentages to trigger RCA")
    crca_pods = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=[], label='Crca Containers', required=True, help_text="Select containers to perform CRCA on if threshold is triggered (can be any number)")


class UploadCGNNTrainDataForm(forms.Form):
    train_array = forms.FileField(label='Train Array')
    test_array = forms.FileField(label='Test Array')
    anomaly_label_array = forms.FileField(label='Anomaly Label Array')
    anomaly_sequence = forms.BooleanField(label='Does this contain an Anomaly Sequence?', required=False)
    dataset = forms.CharField(label='Name Your Dataset', max_length=100)
    comment = forms.CharField(label='Comment', max_length=255, required=False)
    metrics = forms.MultipleChoiceField(
        choices=[],
        widget=forms.CheckboxSelectMultiple,
        label='Select Metrics'
    )
    ordered_metrics = forms.CharField(widget=forms.HiddenInput(), required=False)
