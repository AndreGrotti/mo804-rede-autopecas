param([string]$Origem, [string]$Modelo, [string]$PastaSvg, [string]$Destino, [string]$Link = "[inserir o link do repositório]")
# Monta o relatório final: troca os marcadores {{SVG_*}} pelas figuras SVG e {{LINK_REPOSITORIO}} pelo link.
$l = [IO.File]::ReadAllLines($Origem, [Text.Encoding]::UTF8)
function SvgOrigem([string]$rotulo) {
    $i = 0; while ($i -lt $l.Length -and -not $l[$i].Contains($rotulo)) { $i++ }
    if ($i -ge $l.Length) { throw "SVG não encontrado: $rotulo" }
    $j = $i; while ($l[$j].Trim() -ne '</svg>') { $j++ }
    return ($l[$i..$j] -join "`n")
}
$script:nsvg = 0
function Enxuga([string]$s) {
    # Sem mudar o desenho: tira ids de grupos (não referenciados) e troca estilos repetidos por classes com prefixo
    # próprio de cada figura (as figuras ficam inline no mesmo HTML; o <style> de uma vale para o documento todo).
    $script:nsvg++; $pre = "z$($script:nsvg)c"
    $s = [regex]::Replace($s, '<g id="[^"]*"', '<g')
    $cont = @{}; foreach ($m in [regex]::Matches($s, ' style="([^"]+)"')) { $v = $m.Groups[1].Value; $cont[$v] = 1 + [int]$cont[$v] }
    $cls = @{}; $i = 0; foreach ($k in $cont.Keys) { if ($cont[$k] -ge 3) { $cls[$k] = "$pre$i"; $i++ } }
    if ($cls.Count -eq 0) { return $s }
    $s = [regex]::Replace($s, ' style="([^"]+)"', [Text.RegularExpressions.MatchEvaluator]{ param($m)
        $v = $m.Groups[1].Value; if ($cls.ContainsKey($v)) { ' class="' + $cls[$v] + '"' } else { $m.Value } })
    $css = ($cls.Keys | ForEach-Object { ".$($cls[$_]){$_}" }) -join ''
    return ([regex]'(<svg[^>]*>)').Replace($s, ('$1<style>' + $css.Replace('$', '$$') + '</style>'), 1)
}
function SvgArq([string]$nome) { return (Enxuga ([IO.File]::ReadAllText((Join-Path $PastaSvg "$nome.svg"), [Text.Encoding]::UTF8))) }
$t = [IO.File]::ReadAllText($Modelo, [Text.Encoding]::UTF8)
$mapa = [ordered]@{
    '{{SVG_F1}}' = (SvgArq 'f1_rede'); '{{SVG_F2}}' = (SvgArq 'f2_intervencao_principal'); '{{SVG_F3}}' = (SvgArq 'f3_intervencao_soSP');
    '{{SVG_F4}}' = (SvgArq 'f4_relacoes'); '{{SVG_F5}}' = (SvgArq 'f5_sementes'); '{{SVG_F6}}' = (SvgArq 'f6_vinculos');
    '{{SVG_F7}}' = (SvgArq 'f7_fronteira'); '{{SVG_F2M}}' = (SvgArq 'f2m_marcos');
    '{{SVG_FA1}}' = (SvgArq 'FA1_sementes_parceiros'); '{{SVG_FA2}}' = (SvgArq 'FA2_sementes_desvios');
    '{{SVG_FAI1}}' = (SvgArq 'FAi1_contrafactual'); '{{SVG_FAI2}}' = (SvgArq 'FAi2_formas');
    '{{SVG_FAI3}}' = (SvgArq 'FAi3_simulacao'); '{{SVG_FAI4}}' = (SvgArq 'FAi4_varredura');
    '{{SVG_FHHI}}' = (SvgArq 'FHHI_carteiras');
    '{{SVG_DECOMP}}' = (Enxuga (SvgOrigem 'aria-label="Decomposição')); '{{SVG_DESTINO}}' = (Enxuga (SvgOrigem 'aria-label="Destino das vendas'));
    '{{LINK_REPOSITORIO}}' = $Link
}
foreach ($k in $mapa.Keys) { if (-not $t.Contains($k)) { throw "Marcador ausente no modelo: $k" }; $t = $t.Replace($k, $mapa[$k]) }
if ($t -match '\{\{[A-Z_0-9]+\}\}') { throw "Sobrou marcador: $($Matches[0])" }
if ($t -match 'data:image/') { throw "Ainda há imagem embutida" }
[IO.File]::WriteAllText($Destino, $t, (New-Object Text.UTF8Encoding($false)))
"Gravado: $Destino ($([math]::Round((Get-Item $Destino).Length/1KB)) KB)"
