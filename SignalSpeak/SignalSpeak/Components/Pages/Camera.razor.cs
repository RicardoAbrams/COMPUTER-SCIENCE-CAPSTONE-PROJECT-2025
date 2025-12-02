using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;

namespace SignalSpeak.Components.Pages
{
    public partial class Camera
    {
        private ElementReference _videoRef;
        private IJSObjectReference? _cameraModule;

        private bool _cameraOn;
        private bool _starting;
        private bool _capturing;
        private string? _status;
        private string? _photoDataUrl;

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                try
                {
                    _cameraModule = await JS.InvokeAsync<IJSObjectReference>(
                        "import", "/js/camera.js");

                    _status = "Módulo JS cargado correctamente.";
                }
                catch (Exception ex)
                {
                    _status = $"Error cargando módulo JS: {ex.Message}";
                }

                StateHasChanged();
            }
        }

        private async Task StartCamera()
        {
            _status = "StartCamera() llamado.";
            StateHasChanged();

            if (_cameraModule is null)
            {
                _status = "Error: módulo JS no cargado.";
                StateHasChanged();
                return;
            }

            _photoDataUrl = null;
            _starting = true;
            StateHasChanged();

            try
            {
                var ok = await _cameraModule.InvokeAsync<bool>("startCamera", _videoRef);

                if (ok)
                {
                    _cameraOn = true;
                    _status = "Cámara activa.";
                }
                else
                {
                    _cameraOn = false;
                    _status = "No se pudo activar.";
                }
            }
            catch (JSException ex)
            {
                _status = $"JSException al iniciar cámara: {ex.Message}";
                _cameraOn = false;
            }
            finally
            {
                _starting = false;
                StateHasChanged();
            }
        }

        private async Task StopCamera()
        {
            if (_cameraModule is null)
            {
                _status = "Error: módulo JS no cargado.";
                StateHasChanged();
                return;
            }

            try
            {
                await _cameraModule.InvokeVoidAsync("stopCamera", _videoRef);
                _cameraOn = false;
                _status = "Cámara detenida.";
            }
            catch (Exception ex)
            {
                _status = $"Error al detener cámara: {ex.Message}";
            }

            StateHasChanged();
        }
    }
}
