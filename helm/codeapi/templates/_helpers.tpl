{{- define "codeapi.labels" }}
app.kubernetes.io/name: {{ include "codeapi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "codeapi.selectorLabels" }}
app.kubernetes.io/name: {{ include "codeapi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "codeapi.name" }}
{{ .Chart.Name }}
{{- end }}

{{- define "codeapi.fullname" }}
{{ include "codeapi.name" . }}-{{ .Release.Name }}
{{- end }}
